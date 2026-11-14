import base64
import json
import os

import httpx
from pydantic import ValidationError

from gemini_review import (
    Advisory,
    INSTRUCTION,
    MAX_CROP_BYTES,
    Review,
    review_prompt,
    validate_advisory,
)
from verification import Comparison


VERDA_ENDPOINT = "https://containers.datacrunch.io/nemotron-3-5-super-vl-b300"
DEFAULT_MODEL = "nemotron-3.5-super-vl"


class InvalidInferenceResponse(ValueError):
    pass


def nemotron_key() -> str:
    return os.getenv("NVIDIA_NEMOTRON_KEY", "").strip() or os.getenv("NEMOTRON_API_KEY", "").strip()


def nemotron_config() -> tuple[str, str] | None:
    key = nemotron_key()
    model = os.getenv("NEMOTRON_MODEL", "").strip() or DEFAULT_MODEL
    return (key, model) if key else None


def nemotron_client(key: str) -> httpx.Client:
    return httpx.Client(
        base_url=VERDA_ENDPOINT,
        headers={"Authorization": f"Bearer {key}"},
        timeout=httpx.Timeout(90, connect=10),
        follow_redirects=False,
    )


def discover_model(client: httpx.Client) -> str:
    response = client.get("/v1/models")
    response.raise_for_status()
    model_ids = [entry["id"] for entry in response.json()["data"]]
    if len(model_ids) != 1 or not isinstance(model_ids[0], str) or not model_ids[0].strip():
        raise ValueError("Specify a served model with --model.")
    return model_ids[0]


def ask_nemotron(client: httpx.Client, model: str, crop: bytes, prompt: str, comparison: Comparison) -> Advisory:
    encoded_crop = base64.b64encode(crop).decode("ascii")
    schema = json.dumps(Advisory.model_json_schema(), ensure_ascii=False)
    providers = {item.provider for item in comparison.disputed_evidence}
    local_instruction = (
        " Favor an analyzer only if it has a supplied candidate ID and the visible marking supports it. "
        "Return UNRESOLVED if the crop shows only an opening or the visible marking is ambiguous. "
        "For UNRESOLVED, cite any supplied candidate IDs whose markings you specifically describe."
    )
    if providers == {"ml"}:
        local_instruction += " The only possible supported verdict in this crop is SUPPORTS_ML."
    elif providers == {"deterministic"}:
        local_instruction += " The only possible supported verdict in this crop is SUPPORTS_DETERMINISTIC."
    response = client.post("/v1/chat/completions", json={
        "model": model,
        "messages": [
            {"role": "system", "content": f"{INSTRUCTION}{local_instruction} Respond with a JSON object of this schema: {schema}"},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_crop}"}},
                {"type": "text", "text": prompt},
            ]},
        ],
        "temperature": 0,
        "max_tokens": 4096,
        "reasoning_effort": "low",
        "response_format": {"type": "json_object"},
    })
    response.raise_for_status()
    try:
        payload = response.json()
    except ValueError:
        raise InvalidInferenceResponse("Verda returned a success status with a non-JSON body.") from None
    try:
        choice = payload["choices"][0]
        finish_reason = choice["finish_reason"]
        content = choice["message"]["content"]
    except (LookupError, TypeError):
        raise InvalidInferenceResponse("Verda returned no usable chat completion.") from None
    if finish_reason == "length":
        raise InvalidInferenceResponse("Nemotron exhausted its output token budget before finishing.")
    if finish_reason != "stop":
        raise InvalidInferenceResponse("Nemotron stopped without completing its answer.")
    if not isinstance(content, str) or not content.strip():
        raise InvalidInferenceResponse("Nemotron returned an empty final answer.")
    try:
        return validate_advisory(content, comparison)
    except ValidationError as exc:
        if any(error["type"] == "json_invalid" for error in exc.errors(include_input=False)):
            raise InvalidInferenceResponse("Nemotron's final answer was not valid JSON.") from None
        raise InvalidInferenceResponse("Nemotron's final answer did not match the review schema.") from None
    except ValueError as exc:
        raise InvalidInferenceResponse(str(exc)) from None


def review_with_nemotron(
    crop: bytes, bounds: list[int], page_index: int, comparison: Comparison, key: str, model: str,
) -> Review:
    if len(crop) > MAX_CROP_BYTES:
        return Review(status="outside_scope")
    prompt = review_prompt(bounds, page_index, comparison)
    try:
        with nemotron_client(key) as client:
            advisory = ask_nemotron(client, model, crop, prompt, comparison)
    except httpx.HTTPStatusError as exc:
        code = {
            400: "request_rejected", 401: "access_denied", 403: "access_denied",
            404: "model_unavailable", 408: "timeout", 429: "quota", 504: "timeout",
        }.get(exc.response.status_code, "provider_error")
    except httpx.TimeoutException:
        code = "timeout"
    except httpx.RequestError:
        code = "connection"
    except InvalidInferenceResponse as exc:
        return Review(
            status="unavailable", model=model, crop_bbox_xyxy=bounds,
            error_code="invalid_response", detail=str(exc),
        )
    else:
        return Review(status="reviewed", advisory=advisory, model=model, crop_bbox_xyxy=bounds)
    return Review(status="unavailable", model=model, crop_bbox_xyxy=bounds, error_code=code)
