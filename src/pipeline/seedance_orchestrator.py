# src/pipeline/seedance_orchestrator.py
from __future__ import annotations

import os
from typing import Dict, List, Any


def _cost_per_sec() -> Dict[str, float]:
    return {
        "480p": float(os.getenv("COST_480P", "0.015")),
        "720p": float(os.getenv("COST_720P", "0.025")),
        "1080p": float(os.getenv("COST_1080P", "0.060")),
    }


def _resolution() -> str:
    res = os.getenv("TARGET_RESOLUTION", "720p").lower()
    return res if res in {"480p", "720p", "1080p"} else "720p"


def _cap() -> float:
    try:
        return float(os.getenv("MAX_REPLICATE_COST", "0.30"))
    except Exception:
        return 0.30


def _scene_duration(s: Dict[str, Any]) -> float:
    if isinstance(s.get("start_time"), (int, float)) and isinstance(s.get("end_time"), (int, float)):
        d = float(s["end_time"]) - float(s["start_time"])
        return max(0.1, d)
    if isinstance(s.get("duration_hint_sec"), (int, float)):
        return max(0.1, float(s["duration_hint_sec"]))
    return 6.0


def _scene_weight(s: Dict[str, Any]) -> float:
    w = 0.0
    vm = (s.get("visual_mode") or "").lower()
    if vm == "dynamic":
        w += 1.6
    elif vm == "narrative":
        w += 0.6

    spec = (s.get("image_specificity") or "").lower()
    if spec == "high":
        w += 1.0
    elif spec == "medium":
        w += 0.5

    intent = (s.get("intent") or "").lower()
    if intent == "conceptual":
        w += 0.4

    assets = s.get("assets") or []
    if any((a.get("type") or "image") == "video" for a in assets):
        w += 0.8

    # Prefer scenes with stricter thresholds (suggesting stronger guidance)
    thr = s.get("clip_threshold")
    if isinstance(thr, (int, float)) and float(thr) >= 0.35:
        w += 0.2

    return w


def _scene_cost(seconds: float) -> float:
    return seconds * _cost_per_sec()[_resolution()]


def choose_seedance_scenes(scenes: List[Dict[str, Any]]) -> List[int]:
    """
    Select a subset of scenes to promote to Seedance (video) honoring a budget cap.
    Greedy by (weight / cost) ratio, guarantees at least 2 picks if possible.
    """
    if not scenes:
        return []

    cap = _cap()
    res = _resolution()

    scored: List[Dict[str, Any]] = []
    for s in scenes:
        duration = _scene_duration(s)
        # Assume one Seedance clip per scene, capped to 8s for budget realism
        clip_sec = max(1.0, min(8.0, duration))
        cost = _scene_cost(clip_sec)
        weight = _scene_weight(s)
        ratio = weight / max(1e-6, cost)
        scored.append(
            {
                "index": int(s.get("index", len(scored) + 1)),
                "weight": weight,
                "cost": cost,
                "ratio": ratio,
                "clip_sec": clip_sec,
            }
        )

    scored.sort(key=lambda x: (x["ratio"], x["weight"]), reverse=True)

    picks: List[int] = []
    spend = 0.0
    for it in scored:
        if spend + it["cost"] <= cap:
            picks.append(it["index"])
            spend += it["cost"]

    # Ensure minimum coverage even if cap is tiny
    if not picks and scored:
        picks.append(scored[0]["index"])
    if len(picks) == 1 and len(scored) > 1:
        # Try add the next cheapest option to avoid "PowerPoint feel"
        cheap_sorted = sorted(scored[1:], key=lambda x: x["cost"])
        if cheap_sorted:
            picks.append(cheap_sorted[0]["index"])

    return picks
