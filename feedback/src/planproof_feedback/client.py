"""Small OpenAI-compatible transport; provider reasoning is never used as an answer."""

from __future__ import annotations

import json
import time
import uuid
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from .artifacts import write_json


class ProviderError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ProviderError("Provider redirect refused; check the configured endpoint")


class NemotronClient:
    def __init__(self, settings: dict, key: str, *, opener=None, sleep=time.sleep, audit_dir=None):
        self.settings = settings
        self.key = key
        self.opener = opener or urllib.request.build_opener(NoRedirect())
        self.sleep = sleep
        self.audit_dir = Path(audit_dir) if audit_dir else None
        url = urllib.parse.urlsplit(settings["base_url"])
        if url.scheme != "https" or not url.hostname or url.username or url.password:
            raise ValueError("Provider base_url must be an HTTPS URL without credentials")
        if url.query or url.fragment:
            raise ValueError("Provider base_url must not contain a query or fragment")

    def _request(
        self, suffix: str, payload: dict | None = None, *, audit=None
    ) -> tuple[dict, float]:
        request = urllib.request.Request(
            self.settings["base_url"].rstrip("/") + suffix,
            data=json.dumps(payload, allow_nan=False).encode() if payload is not None else None,
            headers={"Authorization": "Bearer " + self.key, "Content-Type": "application/json"},
        )
        start = time.perf_counter()
        if audit:
            audit.mkdir(parents=True, exist_ok=False)
            write_json(
                audit / "request.json",
                {
                    "url": request.full_url,
                    "body": payload,
                    "credential_policy": "Authorization header deliberately excluded",
                },
            )
        for attempt in range(self.settings.get("retries", 2) + 1):
            try:
                with self.opener.open(
                    request, timeout=self.settings.get("timeout_seconds", 180)
                ) as response:
                    raw = response.read(16 * 1024 * 1024 + 1)
                    if audit:
                        try:
                            (audit / f"response-{attempt + 1}.json").write_bytes(
                                raw.replace(self.key.encode(), b"[REDACTED]")
                            )
                            write_json(
                                audit / f"transport-{attempt + 1}.json",
                                {
                                    "http_status": getattr(response, "status", 200),
                                    "elapsed_seconds": time.perf_counter() - start,
                                    "headers": {
                                        key: response.headers[key]
                                        for key in (
                                            "Content-Type",
                                            "Date",
                                            "X-Request-ID",
                                            "Request-ID",
                                        )
                                        if hasattr(response, "headers") and key in response.headers
                                    },
                                },
                            )
                        except OSError:
                            raise ProviderError(
                                "Local audit could not be saved; request will not be repeated"
                            ) from None
                if len(raw) > 16 * 1024 * 1024:
                    raise ProviderError("Provider response exceeds 16 MiB")
                # Apply the same secret exclusion to derived/validated records as to raw journals.
                result = json.loads(raw.replace(self.key.encode(), b"[REDACTED]"))
                if not isinstance(result, dict):
                    raise ProviderError("Provider response must be a JSON object")
                return result, time.perf_counter() - start
            except urllib.error.HTTPError as exc:
                code = exc.code
                if audit:
                    body = exc.read(1024 * 1024)
                    (audit / f"http-error-{attempt + 1}.txt").write_bytes(
                        body.replace(self.key.encode(), b"[REDACTED]")
                    )
                    write_json(audit / f"attempt-{attempt + 1}.json", {"http_status": code})
                exc.close()
                if code in (429, 500, 502, 503, 504) and attempt < self.settings.get("retries", 2):
                    self.sleep(min(2**attempt, 5))
                    continue
                # Never echo an untrusted response body: it may contain the Authorization header.
                raise ProviderError(
                    f"Provider HTTP {code}; check key, model and endpoint", status_code=code
                ) from None
            except (urllib.error.URLError, TimeoutError, OSError):
                if audit:
                    write_json(
                        audit / f"attempt-{attempt + 1}.json",
                        {"status": "connection_failed_or_timeout"},
                    )
                if attempt < self.settings.get("retries", 2):
                    self.sleep(min(2**attempt, 5))
                    continue
                raise ProviderError("Provider connection failed or timed out") from None
            except (json.JSONDecodeError, UnicodeDecodeError):
                raise ProviderError("Provider did not return valid JSON") from None
        raise ProviderError("Provider retry limit reached")

    def check(self) -> dict:
        audit = self.audit_dir / uuid.uuid4().hex if self.audit_dir else None
        data, elapsed = self._request("/models", audit=audit)
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
        audit = self.audit_dir / uuid.uuid4().hex if self.audit_dir else None
        data, elapsed = self._request("/chat/completions", payload, audit=audit)
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
            "audit_path": str(audit) if audit else None,
        }
