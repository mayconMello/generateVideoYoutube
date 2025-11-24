from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import replicate

from src.media.generation.utils import (
    get_replicate_client,
    log_generation,
    resolve_generation_spec,
    safe_json,
    sanitize_output,
)


def generate_image_with_qwen(
    prompt: str,
    *,
    size: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    aspect_ratio: Optional[str] = None,
    enhance_prompt: bool = True,
    max_images: int = 1,
    sequential_image_generation: str = "disabled",
    image_input: Optional[Sequence[str]] = None,
) -> Tuple[Optional[str], dict]:
    spec = resolve_generation_spec()

    payload = {
        "prompt": prompt,
        "size": size or spec.image_size,
        "width": int(width or spec.width),
        "height": int(height or spec.height),
        "aspect_ratio": aspect_ratio or spec.aspect_ratio,
        "enhance_prompt": bool(enhance_prompt),
        "max_images": int(max(1, max_images)),
        "sequential_image_generation": sequential_image_generation,
        "image_input": list(image_input or []),
    }

    debug: dict = dict(payload)
    client = get_replicate_client()
    if client is None:
        debug["error"] = "missing_replicate_token"
        log_generation(f"Qwen skipped: {safe_json(debug)}")
        return None, debug

    try:
        output = client.run("bytedance/seedream-4", input=payload)
        debug["output_raw"] = sanitize_output(output)
    except replicate.exceptions.ModelError as exc:
        debug["error"] = str(exc)
        detail = getattr(exc, "model_error", None)
        if detail:
            debug["model_error"] = safe_json(detail)
        log_generation(f"Qwen model error: {safe_json(debug)}")
        return None, debug
    except Exception as exc:
        debug["error"] = repr(exc)
        log_generation(f"Qwen exception: {safe_json(debug)}")
        return None, debug

    url = _extract_url(output)
    if url:
        debug["resolved_url"] = url
        return url, debug

    debug["error"] = "no_url_in_output"
    log_generation(f"Seedream output missing URL: {safe_json(debug)}")
    return None, debug


def _extract_url(output: Any) -> Optional[str]:
    if isinstance(output, str) and output.startswith("http"):
        return output
    if hasattr(output, "url"):
        url_attr = getattr(output, "url")
        if callable(url_attr):
            try:
                val = url_attr()
                if isinstance(val, str) and val.startswith("http"):
                    return val
            except Exception:
                pass
        elif isinstance(url_attr, str) and url_attr.startswith("http"):
            return url_attr
    if isinstance(output, (list, tuple)):
        for item in output:
            url = _extract_url(item)
            if url:
                return url
    if isinstance(output, dict):
        for key in ("url", "uri", "image", "href"):
            val = output.get(key)
            if isinstance(val, str) and val.startswith("http"):
                return val
            if isinstance(val, (dict, list, tuple)):
                nested = _extract_url(val)
                if nested:
                    return nested
    return None
