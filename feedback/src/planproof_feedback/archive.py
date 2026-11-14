"""Bounded, read-only ZIP range access. Never downloads the complete Stoa archive."""

from __future__ import annotations

import io
import re
import time
import urllib.error
import urllib.request
from collections import OrderedDict


class RangeArchive(io.RawIOBase):
    def __init__(self, url: str, *, budget=256 * 1024 * 1024, opener=None):
        if not url.startswith("https://"):
            raise ValueError("Archive URL must use HTTPS")
        self.url = url
        self.budget = budget
        self.downloaded = 0
        self.position = 0
        self.opener = opener or urllib.request.build_opener()
        self.etag = None
        self.size = None
        self.cache = OrderedDict()
        self.block_size = 256 * 1024
        self._fetch(0, 1)

    def _fetch(self, start, count):
        if count <= 0:
            return b""
        if self.downloaded + count > self.budget:
            raise ValueError("Archive range-download budget exceeded")
        headers = {"Range": f"bytes={start}-{start + count - 1}", "Accept-Encoding": "identity"}
        if self.etag:
            headers["If-Match"] = self.etag
        request = urllib.request.Request(self.url, headers=headers)
        for attempt in range(3):
            try:
                with self.opener.open(request, timeout=60) as response:
                    if response.status != 206:
                        raise ValueError("Archive server ignored Range; full download refused")
                    match = re.fullmatch(
                        r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", "")
                    )
                    if not match or tuple(map(int, match.groups()[:2])) != (
                        start,
                        start + count - 1,
                    ):
                        raise ValueError("Archive returned an unexpected byte range")
                    total = int(match.group(3))
                    etag = response.headers.get("ETag")
                    if self.size is not None and (
                        total != self.size or (self.etag and etag != self.etag)
                    ):
                        raise ValueError("Remote archive changed during download")
                    self.size, self.etag = total, etag
                    content = response.read(count + 1)
                    self.downloaded += len(content)
                    if len(content) != count:
                        raise ValueError("Incomplete archive byte range")
                    return content
            except (urllib.error.URLError, TimeoutError, OSError):
                if attempt == 2:
                    raise RuntimeError("Stoa archive range request failed") from None
                time.sleep(2**attempt)

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        target = offset + (0 if whence == 0 else self.position if whence == 1 else self.size)
        if target < 0:
            raise ValueError("Negative ZIP seek")
        self.position = target
        return target

    def read(self, size=-1):
        if size < 0:
            size = self.size - self.position
        size = min(size, max(0, self.size - self.position))
        if not size:
            return b""
        start = self.position
        if size >= self.block_size:
            result = self._fetch(start, size)
        else:
            parts = []
            remaining = size
            while remaining:
                block = self.position // self.block_size * self.block_size
                if block not in self.cache:
                    self.cache[block] = self._fetch(block, min(self.block_size, self.size - block))
                    if len(self.cache) > 64:
                        self.cache.popitem(last=False)
                data = self.cache[block]
                offset = self.position - block
                take = min(remaining, len(data) - offset)
                parts.append(data[offset : offset + take])
                self.position += take
                remaining -= take
            result = b"".join(parts)
        self.position = start + size
        return result
