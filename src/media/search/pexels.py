from __future__ import annotations

import os
from typing import List

from pexels_api import API


def search_images(query: str, *, per_page: int = 15) -> List[str]:
    api_key = os.getenv("PEXELS_API_KEY")
    if not api_key or not query:
        return []

    try:
        client = API(api_key)
        client.search(query, page=1, results_per_page=max(1, min(per_page, 80)))
        photos = client.get_entries() or []
    except Exception:
        return []

    urls: List[str] = []
    for photo in photos:
        original = getattr(photo, "original", None)
        if isinstance(original, str) and original.startswith("http"):
            urls.append(original)
    return urls
