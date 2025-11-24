from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Literal


@dataclass(frozen=True)
class VideoProfile:
    key: str
    label: str
    orientation: Literal["portrait", "landscape"]
    width: int
    height: int
    aspect_ratio: str
    asset_min_resolution: tuple[int, int]
    fps: int
    seedance_resolution: str
    generated_image_size: str
    generated_image_width: int
    generated_image_height: int
    fade_in_sec: float
    fade_out_sec: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "key": self.key,
            "label": self.label,
            "orientation": self.orientation,
            "width": self.width,
            "height": self.height,
            "aspect_ratio": self.aspect_ratio,
            "asset_min_resolution": list(self.asset_min_resolution),
            "fps": self.fps,
            "seedance_resolution": self.seedance_resolution,
            "generated_image_size": self.generated_image_size,
            "generated_image_width": self.generated_image_width,
            "generated_image_height": self.generated_image_height,
            "fade_in_sec": self.fade_in_sec,
            "fade_out_sec": self.fade_out_sec,
        }


_PROFILES: Dict[str, VideoProfile] = {
    "shorts": VideoProfile(
        key="shorts",
        label="YouTube Shorts (9:16)",
        orientation="portrait",
        width=1080,
        height=1920,
        aspect_ratio="9:16",
        asset_min_resolution=(1080, 1920),
        fps=60,
        seedance_resolution="1080p",
        generated_image_size="2K",
        generated_image_width=1080,
        generated_image_height=1920,
        fade_in_sec=0.05,
        fade_out_sec=0.05,
    ),
    "landscape": VideoProfile(
        key="landscape",
        label="Landscape 16:9",
        orientation="landscape",
        width=1920,
        height=1080,
        aspect_ratio="16:9",
        asset_min_resolution=(1920, 1080),
        fps=30,
        seedance_resolution="1080p",
        generated_image_size="2K",
        generated_image_width=1920,
        generated_image_height=1080,
        fade_in_sec=0.05,
        fade_out_sec=0.05,
    ),
}

_ALIASES = {
    "short": "shorts",
    "vertical": "shorts",
    "portrait": "shorts",
    "yt_short": "shorts",
    "shorts": "shorts",
    "normal": "landscape",
    "wide": "landscape",
    "horizontal": "landscape",
    "video": "landscape",
}


def resolve_video_profile(name: str | None = None) -> VideoProfile:
    candidate = (name or os.getenv("VIDEO_FORMAT") or os.getenv("VIDEO_PROFILE") or "shorts").strip().lower()
    mapped = _ALIASES.get(candidate, candidate)
    profile = _PROFILES.get(mapped)
    if profile:
        return profile
    return _PROFILES["shorts"]


def apply_profile_env(profile: VideoProfile) -> None:
    os.environ.setdefault("TARGET_RESOLUTION", profile.seedance_resolution)
    os.environ.setdefault("TARGET_ASPECT_RATIO", profile.aspect_ratio)
    os.environ.setdefault("SEEDREAM_IMAGE_SIZE", profile.generated_image_size)
    os.environ.setdefault("SEEDREAM_IMAGE_WIDTH", str(profile.generated_image_width))
    os.environ.setdefault("SEEDREAM_IMAGE_HEIGHT", str(profile.generated_image_height))


__all__ = ["VideoProfile", "resolve_video_profile", "apply_profile_env"]
