import io
import json
import urllib.error

import pytest

from planproof_feedback.client import NemotronClient, NoRedirect, ProviderError
from planproof_feedback.config import get_key, read_env


class Response(io.BytesIO):
    pass


class Opener:
    def __init__(self, values):
        self.values = iter(values)
        self.requests = []

    def open(self, request, timeout):
        self.requests.append(request)
        value = next(self.values)
        if isinstance(value, Exception):
            raise value
        return Response(json.dumps(value).encode())


SETTINGS = {"base_url": "https://example.test/v1", "model": "nemotron", "max_tokens": 123}


def completion(content="final", finish="stop"):
    return {
        "choices": [
            {
                "message": {"content": content, "reasoning_content": "not the answer"},
                "finish_reason": finish,
            }
        ],
        "usage": {"total_tokens": 7},
    }


def test_openai_request_and_reasoning_is_not_final():
    opener = Opener([completion()])
    client = NemotronClient(SETTINGS, "test-secret", opener=opener)
    result = client.complete([{"role": "user", "content": "hello"}])
    request = opener.requests[0]
    body = json.loads(request.data)
    assert request.full_url == "https://example.test/v1/chat/completions"
    assert request.get_header("Authorization") == "Bearer test-secret"
    assert body["max_tokens"] == 123 and "max_completion_tokens" not in body
    assert result["content"] == "final" and "reasoning_content" not in result


@pytest.mark.parametrize("finish", ["length", "tool_calls", "content_filter", None])
def test_nonfinal_completions_are_explicit_failures(finish):
    with pytest.raises(ProviderError, match="Incomplete"):
        NemotronClient(SETTINGS, "secret", opener=Opener([completion(finish=finish)])).complete([])


def test_empty_final_does_not_substitute_reasoning():
    with pytest.raises(ProviderError, match="no final"):
        NemotronClient(SETTINGS, "secret", opener=Opener([completion(content="")])).complete([])


def test_transient_retry_and_auth_error_redaction():
    retry = urllib.error.HTTPError("https://example.test", 429, "busy", {}, io.BytesIO(b"secret"))
    waits = []
    opener = Opener([retry, completion()])
    assert (
        NemotronClient(SETTINGS, "secret", opener=opener, sleep=waits.append).complete([])[
            "content"
        ]
        == "final"
    )
    assert waits == [1]
    failure = urllib.error.HTTPError(
        "https://example.test", 401, "secret", {}, io.BytesIO(b"secret")
    )
    with pytest.raises(ProviderError) as exc:
        NemotronClient(SETTINGS, "secret", opener=Opener([failure])).complete([])
    assert "secret" not in str(exc.value) and "401" in str(exc.value)


@pytest.mark.parametrize(
    "url",
    ["http://example.test/v1", "https://key@example.test/v1", "https://example.test/v1?key=abc"],
)
def test_bad_provider_url(url):
    with pytest.raises(ValueError):
        NemotronClient({**SETTINGS, "base_url": url}, "secret")


def test_redirect_never_forwards_bearer():
    with pytest.raises(ProviderError, match="redirect"):
        NoRedirect().redirect_request(None, None, 302, "", {}, "https://other.test")


def test_check_requires_advertised_model():
    assert (
        NemotronClient(SETTINGS, "secret", opener=Opener([{"data": [{"id": "nemotron"}]}])).check()[
            "status"
        ]
        == "available"
    )
    with pytest.raises(ProviderError, match="not advertised"):
        NemotronClient(SETTINGS, "secret", opener=Opener([{"data": [{"id": "other"}]}])).check()


def test_env_is_literal_and_environment_wins(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('# example\nVERDA_INFERENCE_KEY="literal-$(no-execution)"\n')
    config = {"env_file": str(env)}
    settings = {"api_key_env": "VERDA_INFERENCE_KEY"}
    monkeypatch.delenv("VERDA_INFERENCE_KEY", raising=False)
    assert read_env(env)["VERDA_INFERENCE_KEY"] == "literal-$(no-execution)"
    assert get_key(config, settings) == "literal-$(no-execution)"
    monkeypatch.setenv("VERDA_INFERENCE_KEY", "env-key")
    assert get_key(config, settings) == "env-key"
    monkeypatch.delenv("VERDA_INFERENCE_KEY")
    env.write_text("VERDA_INFERENCE_KEY=\n")
    with pytest.raises(ValueError, match="Paste your full key"):
        get_key(config, settings)


def test_audit_keeps_request_and_complete_provider_response_without_key(tmp_path):
    client = NemotronClient(
        SETTINGS, "private-key", opener=Opener([completion()]), audit_dir=tmp_path
    )
    result = client.complete([{"role": "user", "content": "generated fixture question"}])
    directory = __import__("pathlib").Path(result["audit_path"])
    request = json.loads((directory / "request.json").read_text())
    response = json.loads((directory / "response-1.json").read_text())
    assert request["body"]["messages"][0]["content"] == "generated fixture question"
    assert response["choices"][0]["message"]["reasoning_content"] == "not the answer"
    assert not any("private-key" in p.read_text() for p in directory.iterdir())
    assert "Authorization" not in request


def test_truncated_response_is_preserved_and_error_body_redacts_key(tmp_path):
    client = NemotronClient(
        SETTINGS, "private-key", opener=Opener([completion(finish="length")]), audit_dir=tmp_path
    )
    with pytest.raises(ProviderError):
        client.complete([])
    assert list(tmp_path.rglob("response-1.json"))
    failure = urllib.error.HTTPError(
        "https://example.test", 401, "key", {}, io.BytesIO(b"Bearer private-key")
    )
    with pytest.raises(ProviderError):
        NemotronClient(
            SETTINGS, "private-key", opener=Opener([failure]), audit_dir=tmp_path
        ).complete([])
    body = next(tmp_path.rglob("http-error-1.txt")).read_text()
    assert "private-key" not in body and "[REDACTED]" in body


def test_unexpected_provider_key_echo_is_also_removed_from_derived_output(tmp_path):
    client = NemotronClient(
        SETTINGS,
        "private-key",
        opener=Opener([completion(content="echo private-key")]),
        audit_dir=tmp_path,
    )
    result = client.complete([])
    assert result["content"] == "echo [REDACTED]"
