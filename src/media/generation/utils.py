from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

import replicate


def get_replicate_client() -> Optional[replicate.Client]:
    token = os.getenv("REPLICATE_API_TOKEN")
    if not token:
        return None
    return replicate.Client(api_token=token)


def log_generation(message: str) -> None:
    print(message)


def safe_json(data) -> str:
    try:
        import json

        return json.dumps(data, ensure_ascii=False)
    except Exception:
        return repr(data)


def sanitize_output(data: Any) -> Any:
    """Convert Replicate objects to JSON-serializable structures."""
    if isinstance(data, (str, int, float, bool)) or data is None:
        return data

    if hasattr(data, "url"):
        url_attr = getattr(data, "url")
        if callable(url_attr):
            try:
                val = url_attr()
                if isinstance(val, str) and val.startswith("http"):
                    return val
            except Exception:
                pass
        elif isinstance(url_attr, str) and url_attr.startswith("http"):
            return url_attr

    if isinstance(data, dict):
        return {str(k): sanitize_output(v) for k, v in data.items()}

    if isinstance(data, (list, tuple, set)):
        return [sanitize_output(item) for item in data]

    if hasattr(data, "to_dict"):
        try:
            return sanitize_output(data.to_dict())
        except Exception:
            pass

    return repr(data)


@dataclass(frozen=True)
class GenerationSpec:
    aspect_ratio: str
    width: int
    height: int
    image_size: str
    resolution: str
    fps: int
    seedance_fps: int


def _env_str(name: str, fallback: str) -> str:
    raw = os.getenv(name)
    if not raw:
        return fallback
    return raw.strip() or fallback


def _env_int(name: str, fallback: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return fallback
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return fallback


def resolve_generation_spec() -> GenerationSpec:
    from src.pipeline.video_profiles import resolve_video_profile

    profile = resolve_video_profile(os.getenv("VIDEO_FORMAT"))
    return GenerationSpec(
        aspect_ratio=_env_str("TARGET_ASPECT_RATIO", profile.aspect_ratio),
        width=_env_int("SEEDREAM_IMAGE_WIDTH", profile.generated_image_width),
        height=_env_int("SEEDREAM_IMAGE_HEIGHT", profile.generated_image_height),
        image_size=_env_str("SEEDREAM_IMAGE_SIZE", profile.generated_image_size),
        resolution=_env_str("TARGET_RESOLUTION", profile.seedance_resolution),
        fps=_env_int("TARGET_FPS", profile.fps),
        seedance_fps=_env_int("SEEDANCE_FPS", 24),
    )
