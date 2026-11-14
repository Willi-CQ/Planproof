"""Small OpenAI-compatible transport; provider reasoning is never used as an answer."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request


class ProviderError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ProviderError("Provider redirect refused; check the configured endpoint")


class NemotronClient:
    def __init__(self, settings: dict, key: str, *, opener=None, sleep=time.sleep):
        self.settings = settings
        self.key = key
        self.opener = opener or urllib.request.build_opener(NoRedirect())
        self.sleep = sleep
        url = urllib.parse.urlsplit(settings["base_url"])
        if url.scheme != "https" or not url.hostname or url.username or url.password:
            raise ValueError("Provider base_url must be an HTTPS URL without credentials")
        if url.query or url.fragment:
            raise ValueError("Provider base_url must not contain a query or fragment")

    def _request(self, suffix: str, payload: dict | None = None) -> tuple[dict, float]:
        request = urllib.request.Request(
            self.settings["base_url"].rstrip("/") + suffix,
            data=json.dumps(payload, allow_nan=False).encode() if payload is not None else None,
            headers={"Authorization": "Bearer " + self.key, "Content-Type": "application/json"},
        )
        start = time.perf_counter()
        for attempt in range(self.settings.get("retries", 2) + 1):
            try:
                with self.opener.open(
                    request, timeout=self.settings.get("timeout_seconds", 180)
                ) as response:
                    raw = response.read(16 * 1024 * 1024 + 1)
                if len(raw) > 16 * 1024 * 1024:
                    raise ProviderError("Provider response exceeds 16 MiB")
                result = json.loads(raw)
                if not isinstance(result, dict):
                    raise ProviderError("Provider response must be a JSON object")
                return result, time.perf_counter() - start
            except urllib.error.HTTPError as exc:
                code = exc.code
                exc.close()
                if code in (429, 500, 502, 503, 504) and attempt < self.settings.get("retries", 2):
                    self.sleep(min(2**attempt, 5))
                    continue
                # Never echo an untrusted response body: it may contain the Authorization header.
                raise ProviderError(
                    f"Provider HTTP {code}; check key, model and endpoint", status_code=code
                ) from None
            except (urllib.error.URLError, TimeoutError, OSError):
                if attempt < self.settings.get("retries", 2):
                    self.sleep(min(2**attempt, 5))
                    continue
                raise ProviderError("Provider connection failed or timed out") from None
            except (json.JSONDecodeError, UnicodeDecodeError):
                raise ProviderError("Provider did not return valid JSON") from None
        raise ProviderError("Provider retry limit reached")

    def check(self) -> dict:
        data, elapsed = self._request("/models")
        models = [x.get("id") for x in data.get("data", []) if isinstance(x, dict)]
        expected = self.settings["model"]
        if expected not in models:
            raise ProviderError("Configured model was not advertised by /models; check model ID")
        return {"status": "available", "model": expected, "latency_seconds": elapsed}

    def complete(self, messages: list[dict]) -> dict:
        payload = {
            "model": self.settings["model"],
            "messages": messages,
            "temperature": self.settings.get("temperature", 0),
            "max_tokens": self.settings.get("max_tokens", 4096),
            "stream": False,
        }
        data, elapsed = self._request("/chat/completions", payload)
        try:
            choice = data["choices"][0]
            content = choice["message"]["content"]
            finish = choice.get("finish_reason")
            if finish != "stop":
                raise ProviderError(f"Incomplete provider answer (finish_reason={finish!r})")
            if not isinstance(content, str) or not content.strip():
                raise ProviderError("Provider returned no final text (reasoning is not an answer)")
        except (KeyError, IndexError, TypeError):
            raise ProviderError("Provider response is missing a final assistant message") from None
        usage = data.get("usage") or {}
        if not isinstance(usage, dict):
            usage = {}
        return {
            "content": content,
            "model": data.get("model", self.settings["model"]),
            "usage": {
                key: usage[key]
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                if isinstance(usage.get(key), int) and usage[key] >= 0
            },
            "latency_seconds": elapsed,
            "finish_reason": finish,
        }
