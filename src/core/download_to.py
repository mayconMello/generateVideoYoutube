# src/core/download_to.py
from __future__ import annotations

import base64
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import requests


def _atomic_write(tmp_path: str, final_path: str) -> None:
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    os.replace(tmp_path, final_path)


def _from_data_uri(data_uri: str) -> bytes:
    # Minimal data URI support: data:[<mediatype>][;base64],<data>
    if not data_uri.startswith("data:"):
        raise ValueError("Not a data URI")
    head, b64 = data_uri.split(",", 1)
    if ";base64" not in head:
        return b64.encode("utf-8")
    return base64.b64decode(b64)


def safe_download_to(
    dst_path: str,
    src: str,
    *,
    timeout: float = 60.0,
    attempts: int = 3,
    max_bytes: Optional[int] = None,
    headers: Optional[dict] = None,
) -> str:
    """
    Robust downloader with retries, size guard and atomic write.
    Supports:
      - http(s) URLs
      - file paths
      - data: URIs (base64)

    Returns dst_path on success; raises on failure.
    """
    dst_path = str(dst_path)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    # Local file copy path
    if os.path.exists(src) and os.path.isfile(src):
        tmp = tempfile.NamedTemporaryFile(delete=False, dir=os.path.dirname(dst_path))
        tmp.close()
        shutil.copyfile(src, tmp.name)
        _atomic_write(tmp.name, dst_path)
        return dst_path

    # data: URI
    if isinstance(src, str) and src.startswith("data:"):
        data = _from_data_uri(src)
        if max_bytes is not None and len(data) > max_bytes:
            raise ValueError("Payload exceeds max_bytes")
        tmp = tempfile.NamedTemporaryFile(delete=False, dir=os.path.dirname(dst_path))
        with open(tmp.name, "wb") as f:
            f.write(data)
        _atomic_write(tmp.name, dst_path)
        return dst_path

    # HTTP(S)
    exc: Optional[Exception] = None
    for _ in range(max(1, attempts)):
        try:
            with requests.get(src, stream=True, timeout=timeout, headers=headers or {}) as r:
                r.raise_for_status()
                content_length = r.headers.get("Content-Length")
                if max_bytes is not None and content_length and content_length.isdigit():
                    if int(content_length) > max_bytes:
                        raise ValueError("Remote file exceeds max_bytes")
                tmp = tempfile.NamedTemporaryFile(delete=False, dir=os.path.dirname(dst_path))
                size = 0
                with open(tmp.name, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 15):
                        if not chunk:
                            continue
                        f.write(chunk)
                        size += len(chunk)
                        if max_bytes is not None and size > max_bytes:
                            raise ValueError("Downloaded size exceeds max_bytes")
                _atomic_write(tmp.name, dst_path)
                return dst_path
        except Exception as e:
            exc = e
    if exc:
        raise exc
    return dst_path
