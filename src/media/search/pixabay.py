from __future__ import annotations

import os
from typing import Iterable, List, Optional

import requests

DEFAULT_DENY_TERMS = ["wallpaper", "illustration", "clipart", "render", "ai"]
VIDEO_FORMAT = os.getenv("VIDEO_FORMAT", "shorts") or "shorts"


def _build_query(query: str, deny_terms: Iterable[str]) -> str:
    fragments = [query.strip()]
    for term in deny_terms:
        t = term.strip()
        if not t:
            continue
        if t.startswith("-"):
            fragments.append(t)
        else:
            fragments.append(f"-{t}")
    return " ".join(fragments)


def search_images(
    query: str,
    *,
    deny_terms: Optional[Iterable[str]] = None,
    per_page: int = 12,
    lang: str = "en",
) -> List[str]:
    """
    Query Pixabay for images that match the provided query and return direct URLs.
    """
    api_key = os.getenv("PIXABAY_API_KEY")
    if not api_key or not query:
        return []

    deny = list(DEFAULT_DENY_TERMS)
    if deny_terms:
        for term in deny_terms:
            if term and term not in deny:
                deny.append(term)

    params = {
        "key": api_key,
        "q": query,
        "image_type": "photo",
        "orientation": "vertical" if VIDEO_FORMAT == "shorts" else "horizontal",
        "safesearch": "true",
        "lang": lang,
        "per_page": max(3, min(int(per_page), 50)),
        # Filter for high resolution images to avoid rejections
        "min_width": 1080,  # Minimum width for vertical videos (shorts format)
        "min_height": 1920,  # Minimum height for vertical videos (shorts format)
    }

    try:
        resp = requests.get("https://pixabay.com/api/", params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json() or {}
    except requests.RequestException as e:
        print(e)
        return []

    hits = data.get("hits") or []
    urls: List[str] = []
    for hit in hits:
        if not isinstance(hit, dict):
            continue
        link = hit.get("largeImageURL") or hit.get("fullHDURL") or hit.get("webformatURL")
        if isinstance(link, str) and link.startswith("http"):
            urls.append(link)

    return urls
