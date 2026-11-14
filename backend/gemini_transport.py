"""Bounded Gemini recovery with cancellable I/O and sanitized attempt telemetry."""

from __future__ import annotations

import asyncio
import hashlib
import math
import random
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import httpx
from google import genai
from google.genai import errors, types


@dataclass(frozen=True)
class RetryPolicy:
    attempts: int = 3
    attempt_seconds: float = 90
    turn_seconds: float = 240
    max_requests: int = 12
    initial_backoff: float = 1
    max_backoff: float = 8


class ProviderFailure(RuntimeError):
    def __init__(self, info):
        # Deliberately do not retain exception text, request headers or bodies.
        self.info = info
        super().__init__(info["error_code"])


def response_candidate(response):
    candidates = response.candidates or []
    if not candidates:
        block = getattr(
            getattr(response, "prompt_feedback", None), "block_reason", None
        )
        reason = getattr(block, "value", block)
        if reason is not None and reason not in {r.value for r in types.BlockedReason}:
            reason = "UNKNOWN"
        raise ProviderFailure(
            {
                "error_code": "EmptyResponse",
                "category": "blocked_response" if reason else "incomplete_response",
                "finish_reason": reason,
                "retryable": not bool(reason),
            }
        )
    candidate = candidates[0]
    reason = getattr(candidate.finish_reason, "value", candidate.finish_reason)
    if reason is not None and reason not in {r.value for r in types.FinishReason}:
        reason = "UNKNOWN"
    if reason != "STOP" or not candidate.content or not candidate.content.parts:
        recoverable = reason in (
            None,
            "STOP",
            "MAX_TOKENS",
            "MALFORMED_FUNCTION_CALL",
            "UNEXPECTED_TOOL_CALL",
            "OTHER",
            "FINISH_REASON_UNSPECIFIED",
        )
        raise ProviderFailure(
            {
                "error_code": "IncompleteResponse",
                "category": "incomplete_response"
                if recoverable
                else "blocked_response",
                "finish_reason": reason,
                "retryable": recoverable,
            }
        )
    return candidate


def retry_after(exc):
    response = getattr(exc, "response", None)
    raw = response.headers.get("retry-after") if response is not None else None
    if not raw:
        return None
    try:
        seconds = float(raw)
    except (ValueError, TypeError):
        try:
            seconds = (
                parsedate_to_datetime(raw) - datetime.now(timezone.utc)
            ).total_seconds()
        except (ValueError, TypeError, OverflowError):
            return None
    return max(0.0, seconds) if math.isfinite(seconds) else None


def failure_info(exc):
    if isinstance(exc, ProviderFailure):
        return dict(exc.info)
    info = {
        "error_code": type(exc).__name__,
        "retryable": False,
        "category": "client_error",
    }
    if isinstance(exc, errors.APIError):
        code = exc.code if isinstance(exc.code, int) else None
        info.update(
            http_status=code,
            retryable=code in (408, 429, 500, 502, 503, 504),
            category="rate_limit"
            if code == 429
            else "server_error"
            if code and code >= 500
            else "request_error",
        )
        delay = retry_after(exc)
        if delay is not None:
            info["retry_after_seconds"] = round(delay, 3)
    elif isinstance(exc, (TimeoutError, httpx.TimeoutException)):
        info.update(category="timeout", retryable=True)
    elif isinstance(
        exc, (httpx.NetworkError, httpx.RemoteProtocolError, httpx.ProxyError)
    ):
        info.update(category="connection", retryable=True)
    return info


class ProviderGate:
    """Process-local concurrency cap and short cooldown for repeated outages."""

    def __init__(self):
        self.slots = threading.BoundedSemaphore(2)
        self.lock = threading.Lock()
        self.failures = 0
        self.until = 0.0

    def check(self):
        with self.lock:
            delay = self.until - time.monotonic()
        if delay > 0:
            raise ProviderFailure(
                {
                    "error_code": "ProviderCoolingDown",
                    "category": "cooldown",
                    "retryable": True,
                    "retry_after_seconds": round(delay, 3),
                }
            )

    def record(self, info=None):
        with self.lock:
            if info is None:
                self.failures = 0
                # Keep any Retry-After imposed by another concurrent request.
                return
            if info["category"] not in (
                "connection",
                "timeout",
                "server_error",
                "rate_limit",
            ):
                return
            self.failures += 1
            delay = info.get("retry_after_seconds", 0)
            if self.failures >= 4:
                delay = max(delay, 20)
                self.failures = 0
            self.until = max(self.until, time.monotonic() + delay)


_gates = {}
_gates_lock = threading.Lock()


def shared_gate(key, model):
    identity = hashlib.sha256((key + "\0" + model).encode()).digest()
    with _gates_lock:
        if identity not in _gates:
            if len(_gates) >= 32:
                _gates.pop(next(iter(_gates)))
            _gates[identity] = ProviderGate()
        return _gates[identity]


class ResilientGeminiTransport:
    def __init__(self, key, model, *, instruction, policy=None, gate=None):
        self.model, self._key, self.instruction = model, key, instruction
        self.policy = policy or RetryPolicy()
        self.gate = gate or shared_gate(key, model)
        self.events = []
        self.request_count = 0
        self.last_attempts = 0
        self.on_attempt = None
        self.phase = "normal"

    def generation_profile(self):
        # Capability checked against this configured event model. Unknown models
        # retain the existing settings rather than receiving unsupported parameters.
        bounded_synthesis = self.model.removeprefix(
            "models/"
        ) == "gemini-4-carbon" and self.phase in ("recovery", "token_repair")
        return {
            "max_output_tokens": 16384
            if bounded_synthesis and self.phase == "token_repair"
            else 8192,
            "thinking_level": "LOW" if bounded_synthesis else None,
        }

    async def _generate_async(self, history, tools, seconds):
        # One event loop/connection per attempt: broken pools are never reused.
        # Explicit httpx selects a single SDK transport, without aiohttp's own retries.
        profile = self.generation_profile()
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(seconds, connect=min(10, seconds))
        ) as http:
            client = genai.Client(
                api_key=self._key,
                vertexai=False,
                http_options=types.HttpOptions(
                    timeout=max(1, int(seconds * 1000)),
                    httpx_async_client=http,
                    retry_options=types.HttpRetryOptions(attempts=1),
                ),
            )
            try:
                async with asyncio.timeout(seconds):
                    return await client.aio.models.generate_content(
                        model=self.model,
                        contents=history,
                        config=types.GenerateContentConfig(
                            system_instruction=self.instruction,
                            tools=[types.Tool(function_declarations=tools)],
                            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                                disable=True
                            ),
                            max_output_tokens=profile["max_output_tokens"],
                            thinking_config=types.ThinkingConfig(
                                thinking_level=profile["thinking_level"]
                            )
                            if profile["thinking_level"]
                            else None,
                        ),
                    )
            finally:
                await client.aio.aclose()
                client.close()

    def _generate(self, history, tools, seconds):
        return asyncio.run(self._generate_async(history, tools, seconds))

    def generate(
        self, history, tools, *, deadline=None, phase="normal", max_attempts=None
    ):
        started = time.monotonic()
        self.phase = phase
        deadline = min(
            started + self.policy.turn_seconds if deadline is None else deadline,
            started + self.policy.turn_seconds,
        )
        attempts = min(max_attempts or self.policy.attempts, self.policy.attempts)
        self.last_attempts = 0
        last = None
        for index in range(attempts):
            remaining = deadline - time.monotonic()
            if remaining <= 0.05 or self.request_count >= self.policy.max_requests:
                raise ProviderFailure(
                    {
                        "error_code": "ProviderBudgetExhausted",
                        "category": "budget",
                        "retryable": False,
                    }
                )
            self.gate.check()
            if not self.gate.slots.acquire(timeout=remaining):
                raise ProviderFailure(
                    {
                        "error_code": "ProviderQueueTimeout",
                        "category": "budget",
                        "retryable": True,
                    }
                )
            response = None
            try:
                self.gate.check()
                remaining = deadline - time.monotonic()
                if remaining <= 0.05:
                    raise ProviderFailure(
                        {
                            "error_code": "ProviderBudgetExhausted",
                            "category": "budget",
                            "retryable": False,
                        }
                    )
                self.request_count += 1
                self.last_attempts += 1
                seconds = min(self.policy.attempt_seconds, remaining)
                t0 = time.monotonic()
                event = {
                    "attempt": self.request_count,
                    "phase": phase,
                    "timeout_seconds": round(seconds, 3),
                    **self.generation_profile(),
                }
                try:
                    response = self._generate(history, tools, seconds)
                    response_candidate(response)
                except Exception as exc:
                    info = failure_info(exc)
                    event.update(info)
                    last = ProviderFailure(info)
                    self.gate.record(info)
                else:
                    self.gate.record()
                    event["status"] = "ok"
                    last = None
                finally:
                    event["seconds"] = round(time.monotonic() - t0, 3)
                    if response is not None and response.usage_metadata:
                        event["usage"] = response.usage_metadata.model_dump(mode="json")
                    self.events.append(event)
                    if self.on_attempt:
                        self.on_attempt()
            finally:
                self.gate.slots.release()
            if last is None:
                return response
            # Incomplete/blocked output gets explicit host handling, not blind repeats.
            if (
                not last.info["retryable"]
                or last.info["category"] in ("incomplete_response", "blocked_response")
                or index == attempts - 1
            ):
                raise last
            delay = max(
                self.policy.initial_backoff * 2**index + random.uniform(0, 0.5),
                last.info.get("retry_after_seconds", 0),
            )
            if "retry_after_seconds" not in last.info:
                delay = min(delay, self.policy.max_backoff)
            # Never shorten Retry-After to fit a deadline or retry early.
            if delay + 0.05 >= deadline - time.monotonic():
                raise last
            event["retry_delay_seconds"] = round(delay, 3)
            time.sleep(delay)
        raise last  # pragma: no cover

    def close(self):
        # Each attempt closes both SDK clients, including on cancellation.
        pass
