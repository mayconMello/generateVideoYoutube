from __future__ import annotations

from typing import Iterable, Optional
from urllib.parse import urlparse


def is_allowed_domain(url: str, whitelist: Iterable[str]) -> bool:
    """
    Check if the URL belongs to a whitelisted domain (suffix match).
    """
    if not url:
        return False
    domain = urlparse(url).netloc.lower()
    for allowed in whitelist:
        if not allowed:
            continue
        allowed = allowed.lower().strip()
        if domain.endswith(allowed):
            return True
    return False


def extract_license_if_available(meta: Optional[dict]) -> Optional[str]:
    """
    Extract a human-readable license string from metadata dictionaries returned by APIs.
    """
    if not isinstance(meta, dict):
        return None

    for key in ("license", "licenses", "attribution", "copyright"):
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list):
            joined = ", ".join(str(v).strip() for v in value if str(v).strip())
            if joined:
                return joined
    return None
