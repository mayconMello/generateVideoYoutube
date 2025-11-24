from __future__ import annotations

import math
import os
import shutil
import tempfile
from typing import Optional, Tuple

import replicate
import requests

from src.media.generation.utils import (
    get_replicate_client,
    log_generation,
    resolve_generation_spec,
    sanitize_output,
)


def _normalize_source(image: str) -> Optional[str]:
    if not image:
        return None
    if image.startswith("http"):
        return image
    if os.path.exists(image):
        return image
    return None


def _prepare_image_file(source: str) -> Optional[tuple[str, object]]:
    if source.startswith("http"):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        tmp.close()
        try:
            with requests.get(source, stream=True, timeout=45) as resp:
                resp.raise_for_status()
                with open(tmp.name, "wb") as fh:
                    shutil.copyfileobj(resp.raw, fh)
        except Exception as exc:
            log_generation(f"Seedance keyframe download failed: {exc!r}")
            try:
                os.remove(tmp.name)
            except OSError:
                pass
            return None
        return tmp.name, open(tmp.name, "rb")

    try:
        return "", open(source, "rb")
    except Exception as exc:
        log_generation(f"Seedance keyframe open failed: {exc!r}")
        return None


def generate_video_with_seedance(
    image: str,
    prompt: str,
    *,
    duration: float,
    fps: Optional[int] = None,
    resolution: Optional[str] = None,
    aspect_ratio: Optional[str] = None,
) -> Tuple[Optional[str], dict]:
    spec = resolve_generation_spec()
    fps_value = int(max(1, fps if fps is not None else spec.seedance_fps))
    resolution_value = resolution or spec.resolution
    aspect_ratio_value = aspect_ratio or spec.aspect_ratio
    debug: dict = {
        "prompt": prompt,
        "requested_duration": duration,
        "fps": fps_value,
        "resolution": resolution_value,
        "aspect_ratio": aspect_ratio_value,
    }
    sanitized_duration = max(2, int(math.ceil(duration or 0)))
    debug["duration"] = sanitized_duration
    client = get_replicate_client()
    if client is None:
        debug["error"] = "missing_replicate_token"
        log_generation(f"Seedance skipped: {debug}")
        return None, debug

    normalized = _normalize_source(image)
    if normalized is None:
        debug["error"] = "invalid_reference_image"
        log_generation(f"Seedance invalid reference: {debug}")
        return None, debug

    prepared = _prepare_image_file(normalized)
    if prepared is None:
        debug["error"] = "prepare_image_failed"
        log_generation(f"Seedance could not prepare keyframe: {debug}")
        return None, debug

    tmp_path, handle = prepared
    debug["reference_path"] = normalized

    try:
        output = client.run(
            "bytedance/seedance-1-pro-fast",
            input={
                "fps": fps_value,
                "image": handle,
                "prompt": prompt,
                "duration": sanitized_duration,
                "resolution": resolution_value,
                "aspect_ratio": aspect_ratio_value,
                "camera_fixed": False,
            },
        )
        debug["output_raw"] = sanitize_output(output)
    except replicate.exceptions.ModelError as exc:
        debug["error"] = str(exc)
        detail = getattr(exc, "model_error", None)
        if detail:
            debug["model_error"] = detail
        log_generation(f"Seedance model error: {debug}")
        return None, debug
    except Exception as exc:
        debug["error"] = repr(exc)
        log_generation(f"Seedance exception: {debug}")
        return None, debug
    finally:
        try:
            handle.close()
        except Exception:
            pass
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    url = _extract_url(output)
    if url:
        debug["resolved_url"] = url
        return url, debug

    debug["error"] = "no_url_in_output"
    log_generation(f"Seedance output missing URL: {debug}")
    return None, debug


def _extract_url(output) -> Optional[str]:
    if isinstance(output, str) and output.startswith("http"):
        return output
    if hasattr(output, "url") and isinstance(getattr(output, "url"), str):
        return getattr(output, "url")
    if isinstance(output, (list, tuple)):
        for item in output:
            url = _extract_url(item)
            if url:
                return url
    if isinstance(output, dict):
        for key in ("url", "uri", "video", "href"):
            val = output.get(key)
            url = _extract_url(val)
            if url:
                return url
    return None
