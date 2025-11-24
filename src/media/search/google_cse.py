from __future__ import annotations

import os
from typing import Iterable, List, Optional
from urllib.parse import urlparse

import requests

DEFAULT_DENY_TERMS = ["wallpaper", "illustration", "ai", "concept", "3d", "render"]


def _get_credentials() -> tuple[Optional[str], Optional[str]]:
    api_key = (
            os.getenv("GOOGLE_API_KEY")
            or os.getenv("GOOGLE_CSE_API_KEY")
            or os.getenv("GOOGLE_CSE_KEY")
    )
    cx = (
            os.getenv("GOOGLE_CSE_ID")
            or os.getenv("GOOGLE_CSE_CX")
            or os.getenv("GOOGLE_SEARCH_ENGINE_ID")
    )
    return api_key, cx


def _normalize_query(query: str, deny_terms: Iterable[str]) -> str:
    base = query.strip()
    for term in deny_terms:
        t = term.strip()
        if not t:
            continue
        if t.startswith("-"):
            base += f" {t}"
        else:
            base += f" -{t}"
    return base


def _prioritize_domains(urls: List[str], whitelist: Iterable[str]) -> List[str]:
    wl = [d.strip().lower() for d in whitelist if d and d.strip()]
    if not wl:
        return urls

    def score(url: str) -> tuple[int, str]:
        domain = urlparse(url).netloc.lower()
        for idx, allowed in enumerate(wl):
            if domain.endswith(allowed):
                return (0, f"{idx:02d}-{url}")
        return (1, url)

    return sorted(urls, key=score)


def search_images(
        query: str,
        *,
        domains_whitelist: Optional[Iterable[str]] = None,
        deny_terms: Optional[Iterable[str]] = None,
        num: int = 8,
) -> List[str]:
    """
    Perform a Google CSE image search and return direct image URLs.
    """
    api_key, cx = _get_credentials()
    if not api_key or not cx or not query:
        return []

    deny = list(DEFAULT_DENY_TERMS)
    if deny_terms:
        for term in deny_terms:
            if term and term not in deny:
                deny.append(term)

    params = {
        "key": api_key,
        "cx": cx,
        "q": query,
        "searchType": "image",
        "imgType": "photo",
        "imgSize": "xxlarge",
        "imgAspectRatio": "tall",
        "rights": "cc_publicdomain,cc_attribute",
        "safe": "active",
        "num": 10,
    }

    try:
        resp = requests.get("https://www.googleapis.com/customsearch/v1", params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json() or {}
    except requests.RequestException:
        return []

    items = data.get("items") or []
    urls: List[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        link = item.get("link")
        if not isinstance(link, str) or not link.startswith("http"):
            continue
        urls.append(link)

    if domains_whitelist:
        urls = _prioritize_domains(urls, domains_whitelist)

    return urls[:num]
