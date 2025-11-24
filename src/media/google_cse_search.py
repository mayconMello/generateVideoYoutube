from __future__ import annotations

import os
import time
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional

import requests


@dataclass(frozen=True)
class GoogleImageResult:
    url: str
    width: Optional[int] = None
    height: Optional[int] = None
    mime: Optional[str] = None
    context_url: Optional[str] = None
    title: Optional[str] = None
    display_link: Optional[str] = None
    source: str = "google"


def _env_default(key: str, fallback: Optional[str] = None) -> Optional[str]:
    val = os.getenv(key)
    return val if (val and val.strip()) else fallback


def _coerce_int(v, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(v) if v is not None else default
    except Exception:
        return default


def _intent_params(intent: str) -> Dict[str, str]:
    i = (intent or "factual").strip().lower()
    common = {
        "searchType": "image",
        "safe": "active",
        "imgSize": "xlarge",
    }
    if i == "conceptual":
        return {**common, "imgType": "photo"}
    return {**common, "imgType": "photo"}


def _build_params(
    query: str,
    intent: str,
    api_key: str,
    cx: str,
    lang_restrict: Optional[str],
    num: int,
    start: int,
    rights: Optional[str],
    site_filter: Optional[Iterable[str]],
) -> Dict[str, str]:
    q = query.strip()
    if site_filter:
        sites = [s.strip() for s in site_filter if s and s.strip()]
        if sites:
            q = f"({q}) " + " ".join(f"site:{s}" for s in sites)

    params = {
        "key": api_key,
        "cx": cx,
        "q": q,
        "num": str(max(1, min(10, int(num)))),
        "start": str(max(1, start)),
        **_intent_params(intent),
    }
    if lang_restrict:
        params["lr"] = lang_restrict
    if rights:
        params["rights"] = rights
    return params


def _extract_items(data: dict) -> List[GoogleImageResult]:
    items = data.get("items") or []
    out: List[GoogleImageResult] = []
    for it in items:
        link = it.get("link")
        if not isinstance(link, str) or not link.startswith("http"):
            continue
        img = (it.get("image") or {}) if isinstance(it.get("image"), dict) else {}
        width = _coerce_int(img.get("width"))
        height = _coerce_int(img.get("height"))
        mime = it.get("mime") if isinstance(it.get("mime"), str) else None
        ctx = img.get("contextLink") if isinstance(img.get("contextLink"), str) else None
        title = it.get("title") if isinstance(it.get("title"), str) else None
        dlink = it.get("displayLink") if isinstance(it.get("displayLink"), str) else None
        out.append(
            GoogleImageResult(
                url=link,
                width=width,
                height=height,
                mime=mime,
                context_url=ctx,
                title=title,
                display_link=dlink,
            )
        )
    return out


def _filter_min_resolution(
    results: List[GoogleImageResult],
    min_width: int,
    min_height: int,
) -> List[GoogleImageResult]:
    """Orientation-agnostic resolution filter.

    Accept images whose short side >= min(min_width, min_height)
    AND long side >= max(min_width, min_height).
    """
    out: List[GoogleImageResult] = []
    req_short = min(int(min_width), int(min_height))
    req_long = max(int(min_width), int(min_height))
    for r in results:
        w = int(r.width or 0)
        h = int(r.height or 0)
        short_side = min(w, h)
        long_side = max(w, h)
        if short_side >= req_short and long_side >= req_long:
            out.append(r)
    return out


def _dedupe_by_url(results: List[GoogleImageResult]) -> List[GoogleImageResult]:
    seen: set[str] = set()
    unique: List[GoogleImageResult] = []
    for r in results:
        if r.url not in seen:
            seen.add(r.url)
            unique.append(r)
    return unique


def search_google_images_intent(
    query: str,
    intent: str = "factual",
    *,
    api_key: Optional[str] = None,
    cx: Optional[str] = None,
    num: int = 10,
    lang_restrict: Optional[str] = None,
    rights: Optional[str] = None,
    site_filter: Optional[Iterable[str]] = None,
    min_width: int = 960,
    min_height: int = 540,
    timeout: int = 25,
    retries: int = 2,
    backoff_sec: float = 0.8,
    return_dicts: bool = True,
) -> List[dict] | List[GoogleImageResult]:
    api_key = api_key or _env_default("GOOGLE_CSE_API_KEY")
    cx = cx or _env_default("GOOGLE_CSE_CX")
    if not api_key or not cx:
        return []

    target = max(1, int(num))
    collected: List[GoogleImageResult] = []
    start = 1

    session = requests.Session()
    base_url = "https://www.googleapis.com/customsearch/v1"

    while len(collected) < target:
        page_need = min(10, target - len(collected))
        params = _build_params(
            query=query,
            intent=intent,
            api_key=api_key,
            cx=cx,
            lang_restrict=lang_restrict,
            num=page_need,
            start=start,
            rights=rights,
            site_filter=site_filter,
        )

        attempt = 0
        while True:
            attempt += 1
            try:
                resp = session.get(base_url, params=params, timeout=timeout)
                if resp.status_code == 429 or 500 <= resp.status_code < 600:
                    if attempt <= retries:
                        time.sleep(backoff_sec * (2 ** (attempt - 1)))
                        continue
                    return _finalize(collected, min_width, min_height, return_dicts)
                if resp.status_code != 200:
                    return _finalize(collected, min_width, min_height, return_dicts)
                data = resp.json() or {}
                page_items = _extract_items(data)
                if not page_items:
                    return _finalize(collected, min_width, min_height, return_dicts)
                collected.extend(page_items)
                break
            except requests.RequestException:
                if attempt <= retries:
                    time.sleep(backoff_sec * (2 ** (attempt - 1)))
                    continue
                return _finalize(collected, min_width, min_height, return_dicts)

        start += page_need

    return _finalize(collected, min_width, min_height, return_dicts)


def _finalize(
    results: List[GoogleImageResult],
    min_w: int,
    min_h: int,
    return_dicts: bool,
):
    filtered = _filter_min_resolution(results, min_w, min_h)
    unique = _dedupe_by_url(filtered)
    if return_dicts:
        return [asdict(r) for r in unique]
    return unique
