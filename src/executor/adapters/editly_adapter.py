from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from src.integrations.editly_api import render_with_editly_api, wait_for_health


def _resolve_transition(token: Optional[str], fallback_duration: float) -> Optional[Dict[str, float]]:
    """Disable custom transitions to avoid Editly GLSL crashes."""
    return None


def _ken_burns_for_mode(mode: str, zoom_in_speed: float, zoom_out_speed: float) -> Dict[str, float]:
    mode = (mode or "static").lower()
    if mode == "dynamic":
        return {"zoomDirection": "in", "zoomAmount": zoom_in_speed or 0.3}
    if mode == "narrative":
        return {"zoomDirection": "out", "zoomAmount": zoom_out_speed or 0.2}
    return {"zoomDirection": "in", "zoomAmount": 0.15}


def _ken_burns_from_asset(
    asset: Dict,
    visual_mode: str,
    zoom_in_speed: float,
    zoom_out_speed: float,
) -> Dict[str, float]:
    """Resolve Ken Burns configuration prioritizing per-asset overrides."""
    raw_direction = (asset.get("zoomDirection") or "").strip().lower()
    direction = raw_direction if raw_direction in {"in", "out", "left", "right"} else ""

    def _safe_amount(value) -> float | None:
        try:
            amt = float(value)
        except (TypeError, ValueError):
            return None
        if amt <= 0:
            return None
        return round(float(amt), 4)

    amount = _safe_amount(asset.get("zoomAmount"))

    if direction and amount:
        return {"zoomDirection": direction, "zoomAmount": amount}
    if direction:
        # Provide sane default if direction specified but amount missing
        default = _safe_amount(0.1) or 0.1
        return {"zoomDirection": direction, "zoomAmount": default}
    if amount:
        # Amount provided without direction -> default to zoom in
        return {"zoomDirection": "in", "zoomAmount": amount}
    return _ken_burns_for_mode(visual_mode, zoom_in_speed, zoom_out_speed)


class EditlyAdapter:
    """
    Translate a render_plan produced from VideoRecipe into an Editly configuration.
    """

    def render(
        self,
        *,
        render_plan: Dict,
        audio_path: str,
        output_path: str,
        workdir: str,
    ) -> str:
        recipe = render_plan["recipe"]
        policy = recipe["policy"]["render"]
        scenes: List[Dict] = recipe["scenes"]

        # Use the actual audio duration to guarantee final video covers full narration
        audio_duration = _probe_media_duration(Path(audio_path))
        if audio_duration is None:
            try:
                audio_duration = float((recipe.get("audio") or {}).get("duration_sec") or 0.0)
            except Exception:
                audio_duration = 0.0

        profile_info = render_plan.get("video_profile") or recipe.get("video_profile")
        width = int(profile_info.get("width")) if profile_info and profile_info.get("width") else None
        height = int(profile_info.get("height")) if profile_info and profile_info.get("height") else None
        if not width or not height:
            width, height = self._resolve_dimensions(policy.get("preset"))
        fps_source = profile_info.get("fps") if profile_info else None
        fps = int(fps_source or policy.get("fps", 30))

        config = {
            "outPath": str(Path(output_path).resolve()),
            "width": width,
            "height": height,
            "fps": fps,
            "keepSourceAudio": False,
            "audioFilePath": str(Path(audio_path).resolve()),
            "clips": [],
        }

        global_fade_in = float(policy.get("fade_in_sec", 0.0))
        global_fade_out = float(policy.get("fade_out_sec", 0.0))

        # Precompute target clip durations to absorb silence gaps
        # leading_silence: time before first spoken segment
        leading_silence = 0.0
        if scenes:
            try:
                leading_silence = max(0.0, float(scenes[0]["start_time"]))
            except Exception:
                leading_silence = 0.0

        # Build list of (scene, base_duration, gap_after)
        scene_specs: List[Dict[str, object]] = []
        for i, scene in enumerate(scenes):
            try:
                start = float(scene["start_time"])  # type: ignore[index]
                end = float(scene["end_time"])      # type: ignore[index]
            except Exception:
                # Fallback if scene timings are malformed
                start, end = 0.0, 0.0
            base = max(0.1, end - start)
            if i < len(scenes) - 1:
                try:
                    next_start = float(scenes[i + 1]["start_time"])  # type: ignore[index]
                except Exception:
                    next_start = end
                gap_after = max(0.0, next_start - end)
            else:
                # Last scene: account for tail gap to audio end (if known)
                gap_after = max(0.0, (audio_duration or end) - end)
            scene_specs.append({
                "scene": scene,
                "base": base,
                "gap_after": gap_after,
            })

        # Distribute gaps: add leading silence to first clip, inter-scene gaps to the previous clip
        carry_into_next = 0.0
        for i, spec in enumerate(scene_specs):
            extra = 0.0
            if i == 0:
                extra += leading_silence
            # The gap after this scene will be added to THIS scene's clip duration
            extra += float(spec["gap_after"])  # type: ignore[index]
            target_duration = float(spec["base"]) + extra  # type: ignore[index]
            spec["target_duration"] = round(max(0.1, target_duration), 3)

        # Build clips, merging empty-asset scenes into neighbors to preserve total duration
        pending_carry = 0.0
        last_image_ref: Optional[Dict[str, str]] = None
        last_video_ref: Optional[Dict[str, object]] = None
        for i, spec in enumerate(scene_specs):
            scene: Dict = spec["scene"]  # type: ignore[assignment]
            visual_mode = scene.get("visual_mode", "static")

            # Filter assets that actually resolved to a local path
            assets = [a for a in (scene.get("assets") or []) if isinstance(a, dict) and a.get("local_path")]
            if not assets:
                # No visual layers; carry this time into the next non-empty clip
                pending_carry += float(spec["target_duration"])  # type: ignore[index]
                continue

            # Compute clip duration with any carried time from prior empty scenes
            duration = float(spec["target_duration"]) + pending_carry  # type: ignore[index]
            pending_carry = 0.0

            # Determine base asset timing and scale to fill clip duration
            base_total = 0.0
            base_parts: List[float] = []
            for a in assets:
                try:
                    base_d = float(a.get("duration_sec", 0.0))
                except Exception:
                    base_d = 0.0
                base_d = max(0.1, base_d)
                base_parts.append(base_d)
                base_total += base_d
            scale = (duration / base_total) if base_total > 0 else 1.0

            clip = {
                "duration": round(max(0.1, duration), 3),
                "transition": _resolve_transition(scene.get("transition"), max(0.2, global_fade_in)),
                "layers": [],
            }

            time_cursor = 0.0
            for a, base_d in zip(assets, base_parts):
                local_path = a.get("local_path")
                layer_type = a.get("type", "image")
                asset_dur = max(0.1, base_d * scale)
                end_time = min(duration, time_cursor + asset_dur)
                resolved_path = str(Path(local_path).resolve())

                layer = {
                    "type": layer_type,
                    "path": resolved_path,
                    "start": round(time_cursor, 3),
                    "end": round(end_time, 3),
                }

                if layer_type == "image":
                    layer.update(
                        _ken_burns_from_asset(
                            a,
                            visual_mode,
                            policy.get("zoom_in_speed", 0.25),
                            policy.get("zoom_out_speed", 0.2),
                        )
                    )
                    last_image_ref = {
                        "path": resolved_path,
                        "visual_mode": visual_mode,
                    }
                else:
                    last_video_ref = {
                        "path": resolved_path,
                        "duration_hint": round(asset_dur, 3),
                    }

                # For videos, prefer cover mode to avoid pillarboxing
                layer.setdefault("resizeMode", "cover")
                clip["layers"].append(layer)
                time_cursor = end_time

            if clip["layers"]:
                config["clips"].append(clip)

        tail_pad = _maybe_append_tail_padding(
            config["clips"],
            target_duration=audio_duration,
            safety_margin=float(os.getenv("EDITLY_TIMELINE_SAFETY_SEC", "0.6")),
            zoom_in_speed=policy.get("zoom_in_speed", 0.25),
            zoom_out_speed=policy.get("zoom_out_speed", 0.2),
            image_ref=last_image_ref,
            video_ref=last_video_ref,
        )
        if tail_pad > 0:
            print(f"[editly] added {tail_pad:.3f}s tail padding to guard against CTA truncation.")

        if config["clips"]:
            config["clips"][0].pop("transition", None)
            config["clips"][-1].pop("transition", None)

        if not wait_for_health():
            raise RuntimeError("Editly adapter unavailable (health check failed).")

        ok = render_with_editly_api(config)
        output_resolved = Path(output_path).resolve()
        if not ok:
            # Editly container sometimes produces the video even when the API returns a failure status.
            # Accept the render if the file exists and is non-empty to keep the pipeline resilient.
            if output_resolved.exists() and output_resolved.stat().st_size > 0:
                return str(output_resolved)
            raise RuntimeError("Editly render failed.")
        if not output_resolved.exists() or output_resolved.stat().st_size == 0:
            raise RuntimeError("Editly render reported success but output is missing or empty.")
        _remux_to_audio_length(
            video_path=output_resolved,
            audio_path=Path(audio_path),
            audio_duration=audio_duration,
            tolerance=float(os.getenv("EDITLY_AUDIO_SYNC_TOLERANCE", "0.02")),
        )
        return str(output_resolved)

    @staticmethod
    def _resolve_dimensions(preset: Optional[str]) -> tuple[int, int]:
        preset = (preset or "").lower()
        if preset == "1080p":
            return 1920, 1080
        if preset == "480p":
            return 1280, 720
        return 1280, 720


def _probe_media_duration(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    except Exception:
        return None
    if proc.returncode not in (0, 1):
        return None
    try:
        value = float((proc.stdout or "").strip().splitlines()[-1])
    except (IndexError, ValueError):
        return None
    return value if value > 0 else None


def _maybe_append_tail_padding(
    clips: List[Dict],
    *,
    target_duration: Optional[float],
    safety_margin: float,
    zoom_in_speed: float,
    zoom_out_speed: float,
    image_ref: Optional[Dict[str, str]],
    video_ref: Optional[Dict[str, object]],
) -> float:
    if not clips:
        return 0.0
    declared_total = sum(float(c.get("duration", 0.0)) for c in clips)
    desired_min = max(declared_total, float(target_duration or 0.0)) + max(0.0, safety_margin)
    pad = max(0.0, desired_min - declared_total)
    if pad < 0.05:
        return 0.0

    if _append_tail_clip(
        clips,
        duration=pad,
        zoom_in_speed=zoom_in_speed,
        zoom_out_speed=zoom_out_speed,
        image_ref=image_ref,
        video_ref=video_ref,
    ):
        return round(pad, 3)

    # Fallback: stretch the last layer if no assets are available (rare).
    last_clip = clips[-1]
    last_clip["duration"] = round(float(last_clip.get("duration", 0.0)) + pad, 3)
    if last_clip.get("layers"):
        last_layer = last_clip["layers"][-1]
        last_layer["end"] = round(float(last_layer.get("end", last_clip["duration"])) + pad, 3)
    return round(pad, 3)


def _append_tail_clip(
    clips: List[Dict],
    *,
    duration: float,
    zoom_in_speed: float,
    zoom_out_speed: float,
    image_ref: Optional[Dict[str, str]],
    video_ref: Optional[Dict[str, object]],
) -> bool:
    duration = round(duration, 3)
    if duration <= 0:
        return False

    layer: Optional[Dict] = None
    if image_ref:
        layer = {
            "type": "image",
            "path": image_ref["path"],
            "start": 0.0,
            "end": duration,
            "resizeMode": "cover",
        }
        layer.update(
            _ken_burns_for_mode(
                image_ref.get("visual_mode", "static"),
                zoom_in_speed,
                zoom_out_speed,
            )
        )
    elif video_ref:
        cut_from = max(0.0, float(video_ref.get("duration_hint", duration)) - duration)
        layer = {
            "type": "video",
            "path": video_ref["path"],
            "start": 0.0,
            "end": duration,
            "cutFrom": round(cut_from, 3),
            "cutTo": round(cut_from + duration, 3),
            "resizeMode": "cover",
        }

    if not layer:
        return False

    clips.append(
        {
            "duration": duration,
            "transition": None,
            "layers": [layer],
        }
    )
    return True


def _remux_to_audio_length(
    *,
    video_path: Path,
    audio_path: Path,
    audio_duration: Optional[float],
    tolerance: float,
) -> None:
    if not video_path.exists() or not audio_path.exists():
        return

    video_len = _probe_media_duration(video_path)
    audio_len = audio_duration or _probe_media_duration(audio_path)
    if not video_len or not audio_len:
        return

    tolerance = max(0.01, tolerance)
    if abs(video_len - audio_len) <= tolerance:
        return
    if video_len + tolerance < audio_len:
        print(
            f"[editly] warning: rendered video ({video_len:.3f}s) shorter than narration ({audio_len:.3f}s); "
            "remux skipped."
        )
        return

    tmp_path = video_path.with_name(f"{video_path.stem}_synced{video_path.suffix}")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "copy",
        "-shortest",
        str(tmp_path),
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    except Exception as exc:
        print(f"[editly] remux failed: {exc}")
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        return

    if proc.returncode != 0:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        return

    try:
        tmp_path.replace(video_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
