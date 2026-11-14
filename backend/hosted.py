"""Serve Stoa behind HTTP Basic authentication for a shared HTTPS demo."""

import base64
import binascii
import os
from secrets import compare_digest

from starlette.datastructures import Headers
from starlette.responses import PlainTextResponse

from app import app as stoa_app


password = os.environ.get("STOA_ACCESS_PASSWORD", "")
if not password:
    raise RuntimeError("Configure STOA_ACCESS_PASSWORD before starting the hosted app.")
expected = f"stoa:{password}".encode()


async def app(scope, receive, send):
    if scope["type"] != "http":
        await stoa_app(scope, receive, send)
        return
    scheme, _, encoded = Headers(scope=scope).get("authorization", "").partition(" ")
    if scheme.casefold() == "basic":
        try:
            credentials = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError):
            credentials = b""
        if compare_digest(credentials, expected):
            await stoa_app(scope, receive, send)
            return
    response = PlainTextResponse(
        "Stoa demo access is required.",
        status_code=401,
        headers={"WWW-Authenticate": 'Basic realm="Stoa demo"', "Cache-Control": "no-store"},
    )
    await response(scope, receive, send)
