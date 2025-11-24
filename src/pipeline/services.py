from __future__ import annotations

import base64
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable, Union

import whisper_timestamped as whisper
from elevenlabs.client import ElevenLabs


@dataclass
class NarrationAssets:
    audio_path: str
    alignment: Optional[dict] = None


_DEFAULT_OUTPUT_FORMAT = os.getenv("ELEVENLABS_OUTPUT_FORMAT", "mp3_44100_192")
_MIN_SEGMENT_DURATION = 1.0
_SILENCE_NOISE_DB = float(os.getenv("SEGMENTS_SILENCE_DB", "-45"))
_SILENCE_MIN_SEC = float(os.getenv("SEGMENTS_SILENCE_MIN_SEC", "0.25"))
_SILENCE_TOLERANCE_SEC = float(os.getenv("SEGMENTS_SILENCE_TOLERANCE_SEC", "0.05"))

_TAG_ANYWHERE_RE = re.compile(r"\[[^\]]+\]\s*")
_PUNCT_ONLY_RE = re.compile(r"^[\s.?!,;:…-]*$")
_ELLIPSIS_RE = re.compile(r"\.\s*\.\s*\.")
_MULTI_SPACE_RE = re.compile(r"\s+")


def _resolve_text_normalization_mode() -> str:
    value = (os.getenv("ELEVENLABS_TEXT_NORMALIZATION") or "auto").strip().lower()
    return value if value in {"auto", "on", "off"} else "auto"


def _resolve_language_text_normalization_flag() -> Optional[bool]:
    raw = os.getenv("ELEVENLABS_LANGUAGE_TEXT_NORMALIZATION")
    if raw is None:
        return None
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None


def _resolve_language_code() -> Optional[str]:
    raw = os.getenv("ELEVENLABS_LANGUAGE_CODE")
    if raw is None:
        return "pt"
    code = raw.strip()
    if not code:
        return None
    return code.lower()


def _clean_alignment_text(text: str) -> str:
    cleaned = (text or "").replace("\n", " ").strip()
    cleaned = _TAG_ANYWHERE_RE.sub(" ", cleaned)
    cleaned = _MULTI_SPACE_RE.sub(" ", cleaned)
    cleaned = _ELLIPSIS_RE.sub("...", cleaned)
    return cleaned.strip()


def _first_spoken_timestamp(
        buffer: List[Tuple[str, Optional[float], Optional[float]]],
) -> Optional[float]:
    tag_depth = 0
    for ch, start, end in buffer:
        if ch == "[":
            tag_depth += 1
            continue
        if ch == "]" and tag_depth > 0:
            tag_depth -= 1
            continue
        if tag_depth > 0 or ch.isspace():
            continue
        return float(start if start is not None else (end if end is not None else 0.0))
    return None


def _last_spoken_timestamp(
        buffer: List[Tuple[str, Optional[float], Optional[float]]],
) -> Optional[float]:
    tag_depth = 0
    for ch, start, end in reversed(buffer):
        if ch == "]":
            tag_depth += 1
            continue
        if ch == "[" and tag_depth > 0:
            tag_depth -= 1
            continue
        if tag_depth > 0 or ch.isspace():
            continue
        return float(end if end is not None else (start if start is not None else 0.0))
    return None


def _segment_duration(segment: dict) -> float:
    start = float(segment.get("start", 0.0))
    end = float(segment.get("end", start))
    return max(0.0, end - start)


def _maybe_extract_audio_bytes(obj) -> Optional[bytes]:
    """Best-effort extractor for audio bytes from arbitrary SDK objects.

    Checks common attributes seen in ElevenLabs responses.
    """
    if obj is None:
        return None

    if isinstance(obj, (bytes, bytearray)):
        return bytes(obj)

    for attr in ("audio_base_64", "audio_base64", "audio", "mp3_base64", "content"):
        data = getattr(obj, attr, None)
        if isinstance(data, (bytes, bytearray)):
            return bytes(data)
        if isinstance(data, str):

            try:
                return base64.b64decode(data)
            except Exception:
                continue

    to_bytes = getattr(obj, "to_bytes", None)
    if callable(to_bytes):
        try:
            data = to_bytes()
            if isinstance(data, (bytes, bytearray)):
                return bytes(data)
        except Exception:
            pass
    return None


def _is_punctuation_segment(segment: dict) -> bool:
    text = segment.get("clean_text") or _clean_alignment_text(segment.get("text", ""))
    return _PUNCT_ONLY_RE.match(text or "") is not None


def _merge_segment(segments: List[dict], src_idx: int, dest_idx: int, *, prepend: bool) -> None:
    if src_idx == dest_idx or not (0 <= src_idx < len(segments)) or not (0 <= dest_idx < len(segments)):
        return

    src = segments[src_idx]
    dest = segments[dest_idx]

    dest["start"] = min(float(dest.get("start", 0.0)), float(src.get("start", 0.0)))
    dest["end"] = max(float(dest.get("end", 0.0)), float(src.get("end", 0.0)))

    if prepend:
        combined = f"{src.get('text', '').strip()} {dest.get('text', '').strip()}".strip()
    else:
        combined = f"{dest.get('text', '').strip()} {src.get('text', '').strip()}".strip()

    dest["text"] = combined.strip()
    dest["clean_text"] = _clean_alignment_text(dest["text"])

    segments.pop(src_idx)


def _merge_punctuation_segments(segments: List[dict]) -> None:
    i = 0
    while i < len(segments):
        if _is_punctuation_segment(segments[i]):
            if len(segments) == 1:
                break
            if i == 0:
                _merge_segment(segments, i, i + 1, prepend=True)
            else:
                _merge_segment(segments, i, i - 1, prepend=False)
                i -= 1
        else:
            i += 1


def _enforce_min_duration(segments: List[dict], min_duration: float) -> None:
    if min_duration <= 0:
        return

    i = 0
    while i < len(segments):
        if len(segments) <= 1:
            break
        duration = _segment_duration(segments[i])
        if duration >= min_duration:
            i += 1
            continue

        if i == 0:
            _merge_segment(segments, i, i + 1, prepend=True)
        elif i == len(segments) - 1:
            _merge_segment(segments, i, i - 1, prepend=False)
            i -= 1
        else:
            prev_duration = _segment_duration(segments[i - 1])
            next_duration = _segment_duration(segments[i + 1])
            if next_duration < prev_duration:
                _merge_segment(segments, i, i + 1, prepend=True)
            else:
                _merge_segment(segments, i, i - 1, prepend=False)
                i -= 1


def _detect_audio_silences(
        audio_path: Optional[str],
        *,
        noise_db: float,
        min_silence_sec: float,
) -> List[Tuple[float, float]]:
    if not audio_path:
        return []
    path = Path(audio_path)
    if not path.exists():
        return []

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-i",
        str(path),
        "-af",
        f"silencedetect=noise={noise_db}dB:d={min_silence_sec}",
        "-f",
        "null",
        "-",
    ]

    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, text=True)
    except Exception:
        return []

    stderr = proc.stderr or ""
    silences: List[Tuple[float, float]] = []
    current_start: Optional[float] = None
    for line in stderr.splitlines():
        line = line.strip()
        if "silence_start" in line:
            try:
                current_start = float(line.split("silence_start:")[1].strip())
            except (ValueError, IndexError):
                current_start = None
        elif "silence_end" in line and current_start is not None:
            try:
                parts = line.split("silence_end:")[1].split("|")
                end_val = float(parts[0].strip())
                silences.append((current_start, end_val))
            except (ValueError, IndexError):
                pass
            finally:
                current_start = None
    return silences


def _apply_silence_gaps(segments: List[dict], silences: List[Tuple[float, float]], tolerance: float) -> List[dict]:
    if not segments or not silences:
        return segments

    adjusted = [dict(seg) for seg in segments]

    for start, end in silences:
        duration = max(0.0, end - start)
        if duration <= 0.0:
            continue

        prev_idx = None
        for idx, seg in enumerate(adjusted):
            if seg["end"] <= start + tolerance:
                prev_idx = idx
            elif seg["start"] - start > tolerance:
                break

        if prev_idx is None:
            for seg in adjusted:
                seg["start"] += duration
                seg["end"] += duration
            continue

        next_idx = prev_idx + 1
        base_gap = 0.0
        if next_idx < len(adjusted):
            base_gap = max(0.0, adjusted[next_idx]["start"] - adjusted[prev_idx]["end"])
        delta = duration - base_gap
        if delta <= 0.01:
            continue

        adjusted[prev_idx]["end"] += delta
        for idx in range(next_idx, len(adjusted)):
            adjusted[idx]["start"] += delta
            adjusted[idx]["end"] += delta

    for seg in adjusted:
        seg["start"] = round(seg["start"], 3)
        seg["end"] = round(seg["end"], 3)

    return adjusted


def _probe_audio_duration(audio_path: Optional[str]) -> Optional[float]:
    if not audio_path:
        return None
    path = Path(audio_path)
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
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, text=True)
    except Exception:
        return None
    if proc.returncode not in (0, 1):
        return None
    try:
        value_str = (proc.stdout or "").strip().splitlines()[-1]
        value = float(value_str)
    except (IndexError, ValueError):
        return None
    return value if value > 0 else None


def _normalize_segment_timeline(segments: List[dict], target_duration: Optional[float]) -> List[dict]:
    if not segments or not target_duration or target_duration <= 0:
        return segments
    last_end = segments[-1]["end"]
    if last_end <= 0:
        return segments
    diff = target_duration - last_end
    if abs(diff) <= 0.05:
        return segments

    normalized = [dict(seg) for seg in segments]
    new_end = normalized[-1]["end"] + diff
    if new_end <= normalized[-1]["start"]:
        new_end = normalized[-1]["start"] + 0.1
    normalized[-1]["end"] = round(new_end, 3)
    return normalized


def synthesize_narration(
        client: ElevenLabs,
        *,
        text: str,
        base_dir: str,
        voice_id: str,
        model_id: str = "eleven_v3",
        voice_settings: Optional[Dict[str, Union[float, bool]]] = None,
        logger: Optional[callable] = None,
) -> NarrationAssets:
    log = logger or (lambda msg: None)

    request_kwargs: Dict[str, object] = {
        "voice_id": voice_id,
        "model_id": model_id,
        "output_format": _DEFAULT_OUTPUT_FORMAT,
        "text": text,
        "apply_text_normalization": _resolve_text_normalization_mode(),
    }
    language_code = _resolve_language_code()
    if language_code:
        request_kwargs["language_code"] = language_code
    lang_norm = _resolve_language_text_normalization_flag()
    if lang_norm is not None:
        request_kwargs["apply_language_text_normalization"] = lang_norm
    voice_settings_payload: Dict[str, Union[float, bool]] = {
        "stability": 0.5,
        "similarity_boost": 0.75,
        "style": 0.75,
        "use_speaker_boost": True,
        "speed": 1.2,
    }
    request_kwargs["voice_settings"] = voice_settings_payload

    response = client.text_to_speech.convert_with_timestamps(**request_kwargs)

    audio_path = os.path.join(base_dir, "narration_full.mp3")
    audio_bytes = base64.b64decode(response.audio_base_64)
    Path(audio_path).write_bytes(audio_bytes)

    alignment_obj = getattr(response, "normalized_alignment", None) or getattr(response, "alignment", None)
    if not alignment_obj:
        raise RuntimeError("ElevenLabs did not return alignment data.")

    alignment_data = {
        "source": "normalized_alignment" if getattr(response, "normalized_alignment", None) else "alignment",
        "characters": alignment_obj.characters,
        "start_times": alignment_obj.character_start_times_seconds,
        "end_times": alignment_obj.character_end_times_seconds,
        "audio_format": _DEFAULT_OUTPUT_FORMAT,
    }

    log(f"[narration] alignment source: {alignment_data['source']}")
    return NarrationAssets(audio_path=audio_path, alignment=alignment_data)


def build_segments_from_alignment(
        narration_text: str,
        alignment_data: dict,
        *,
        audio_path: Optional[str] = None,
) -> List[dict]:
    chars: List[str] = alignment_data.get("characters") or []
    starts: List[Optional[float]] = alignment_data.get("start_times") or []
    ends: List[Optional[float]] = alignment_data.get("end_times") or []
    if not chars or not starts or not ends:
        raise RuntimeError("Incomplete alignment data.")

    segments: List[dict] = []
    buffer: List[Tuple[str, Optional[float], Optional[float]]] = []
    current_start: Optional[float] = None

    n = len(chars)
    idx = 0
    while idx < n:
        ch = chars[idx]
        ch_start = starts[idx] if idx < len(starts) else None
        ch_end = ends[idx] if idx < len(ends) else None

        if ch == "[":
            if buffer:
                prev_end = _last_spoken_timestamp(buffer)
                seg_s = _first_spoken_timestamp(buffer)
                if seg_s is None:
                    seg_s = current_start if current_start is not None else float(ch_start or 0.0)
                seg_e = prev_end if prev_end is not None else float(ch_end if ch_end is not None else seg_s + 0.5)
                if seg_e <= seg_s:
                    seg_e = seg_s + 0.5
                raw_text = "".join(x[0] for x in buffer)
                segments.append(
                    {
                        "start": float(seg_s),
                        "end": float(seg_e),
                        "text": raw_text,
                        "clean_text": _clean_alignment_text(raw_text),
                    }
                )
            buffer = []
            current_start = None

            j = idx + 1
            while j < n and chars[j] != "]":
                j += 1
            idx = j + 1 if j < n else n
            continue

        if current_start is None and ch_start is not None:
            current_start = float(ch_start)
        buffer.append((ch, ch_start, ch_end))

        is_boundary = (ch in ".?!") or (idx == n - 1)
        if is_boundary:
            seg_s = _first_spoken_timestamp(buffer)
            if seg_s is None:
                seg_s = current_start if current_start is not None else float(ch_start or 0.0)
            seg_e = _last_spoken_timestamp(buffer)
            if seg_e is None:
                seg_e = float(ch_end if ch_end is not None else seg_s + 0.5)
            if seg_e <= seg_s:
                seg_e = seg_s + 0.5
            raw_text = "".join(x[0] for x in buffer)
            segments.append(
                {
                    "start": float(seg_s),
                    "end": float(seg_e),
                    "text": raw_text,
                    "clean_text": _clean_alignment_text(raw_text),
                }
            )
            buffer = []
            current_start = None

        idx += 1

    _merge_punctuation_segments(segments)
    _enforce_min_duration(segments, _MIN_SEGMENT_DURATION)

    normalized: List[dict] = []
    prev_end = 0.0
    for i, seg in enumerate(segments, start=1):
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start + 0.5))
        if start < prev_end:
            start = prev_end
        if end <= start:
            end = start + 0.5
        text = seg.get("clean_text") or _clean_alignment_text(seg.get("text", ""))
        normalized.append(
            {
                "index": i,
                "start": round(start, 3),
                "end": round(end, 3),
                "text": text.strip(),
            }
        )
        prev_end = end

    silences = _detect_audio_silences(
        audio_path,
        noise_db=_SILENCE_NOISE_DB,
        min_silence_sec=_SILENCE_MIN_SEC,
    )
    if silences:
        normalized = _apply_silence_gaps(normalized, silences, tolerance=_SILENCE_TOLERANCE_SEC)

    target_duration = _probe_audio_duration(audio_path) or (normalized[-1]["end"] if normalized else None)
    normalized = _normalize_segment_timeline(normalized, target_duration)

    return normalized


def split_script_into_phrases(script: str) -> List[str]:
    normalized = script.replace("\n", " ").strip()

    parts = re.split(r'(?<=[\.!?…])\s+', normalized)
    return [p.strip() for p in parts if p.strip()]


def apply_script_to_segments(
        segments: List[Dict],
        narration_text: str,
        logger: Optional[Callable[[str], None]] = None,
) -> List[Dict]:
    log = logger or (lambda msg: None)

    phrases = split_script_into_phrases(narration_text)
    if not phrases:
        log("[segments] narration script is empty or could not be split; keeping Whisper text")
        return segments

    log(f"[segments] mapping {len(phrases)} script phrases to {len(segments)} segments")

    n = min(len(segments), len(phrases))

    for i in range(n):
        segments[i]["text"] = phrases[i]

    if len(phrases) > len(segments):
        extra = " ".join(phrases[n:])
        segments[-1]["text"] = (segments[-1]["text"] + " " + extra).strip()
        log(f"[segments] {len(phrases) - len(segments)} extra phrases appended to last segment")

    return segments


def generate_whisper_segments(
        audio_path: str,
        *,
        logger: Optional[callable] = None,
) -> List[dict]:
    log = logger or (lambda msg: None)

    if not audio_path:
        raise RuntimeError("Audio path not provided for Whisper segmentation.")

    model = whisper.load_model("large-v2", device="cpu")
    log("[segments] whisper_timestamped(model=large, language=pt)")

    result = whisper.transcribe(
        model,
        audio_path,
        language="pt",
        temperature=0.0,
        beam_size=1,
        best_of=1
    )

    segments = result.get("segments") or []

    normalized: List[dict] = []
    prev_end = 0.0

    for idx, seg in enumerate(segments, start=1):
        start = float(seg.get("start", prev_end))
        end = float(seg.get("end", start + 0.1))

        if start < prev_end:
            start = prev_end
        if end <= start:
            end = start + 0.1

        text = (seg.get("text") or "").strip()

        normalized.append(
            {
                "index": idx,
                "start": round(start, 3),
                "end": round(end, 3),
                "text": text,
            }
        )

        prev_end = end

    return normalized


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compose_background_music(
        client: ElevenLabs,
        *,
        prompt: str,
        duration_sec: float,
        base_dir: str,
        logger: Optional[callable] = None,
        force_instrumental: bool = True,
) -> Optional[str]:
    """Compose background music following the official Eleven Music quickstart."""
    log = logger or (lambda msg: None)
    music_dir = Path(base_dir)
    music_dir.mkdir(parents=True, exist_ok=True)
    music_path = music_dir / "background_music.mp3"
    plan_path = music_dir / "background_music_plan.json"
    metadata_path = music_dir / "background_music_metadata.json"
    if music_path.exists() and music_path.stat().st_size > 0:
        log(f"[music] reusing existing background music -> {music_path}")
        return str(music_path)
    tail_sec = max(0.0, float(os.getenv("BACKGROUND_MUSIC_TAIL_SEC", "5")))
    duration_padding = 1.5
    target_duration = max(0.1, float(duration_sec) + duration_padding + tail_sec)
    duration_ms = int(target_duration * 1000)
    output_format = os.getenv("ELEVEN_MUSIC_OUTPUT_FORMAT", "mp3_44100_192")

    music_api = getattr(client, "music", None)
    if music_api is None:
        log("[music] ElevenLabs client does not expose 'music' API; skipping composition.")
        return None

    def _write_audio(data: bytes, source: str) -> str:
        music_path.write_bytes(data)
        log(f"[music] background music saved via {source} -> {music_path}")
        return str(music_path)

    def _stream_compose(*, force_allowed: bool = True, **kwargs) -> Optional[bytes]:
        compose_kwargs = {
            "output_format": output_format,
            **kwargs,
        }
        if force_allowed:
            compose_kwargs["force_instrumental"] = force_instrumental

        try:
            stream = music_api.compose(**compose_kwargs)
        except Exception as exc:
            log(f"[music] compose error ({'plan' if kwargs.get('composition_plan') else 'prompt'}): {exc}")
            return None

        if isinstance(stream, (bytes, bytearray)):
            return bytes(stream)

        data = bytearray()
        try:
            for chunk in stream:
                if not chunk:
                    continue
                if isinstance(chunk, (bytes, bytearray)):
                    data.extend(chunk)
                elif isinstance(chunk, str):
                    data.extend(chunk.encode("utf-8"))
                else:
                    try:
                        data.extend(bytes(chunk))
                    except Exception:
                        continue
        except Exception as exc:
            log(f"[music] compose stream failure: {exc}")
            return None
        return bytes(data) if data else None

    plan_obj = None
    try:
        plan_obj = music_api.composition_plan.create(
            prompt=prompt,
            music_length_ms=duration_ms,
        )
        log("[music] composition plan created.")
        try:
            if hasattr(plan_obj, "model_dump_json"):
                plan_path.write_text(plan_obj.model_dump_json(indent=2), encoding="utf-8")
            elif hasattr(plan_obj, "json"):
                plan_path.write_text(json.dumps(plan_obj.json, ensure_ascii=False, indent=2), encoding="utf-8")
            else:
                plan_path.write_text(json.dumps(getattr(plan_obj, "__dict__", {}), ensure_ascii=False, indent=2),
                                     encoding="utf-8")
        except Exception:
            pass
    except Exception as exc:
        log(f"[music] composition plan creation failed; will compose directly from prompt: {exc}")
        plan_obj = None

    if plan_obj:
        audio_bytes = _stream_compose(force_allowed=False, composition_plan=plan_obj)
        if audio_bytes:
            return _write_audio(audio_bytes, "composition_plan")

    audio_bytes = _stream_compose(
        prompt=prompt,
        music_length_ms=duration_ms,
    )
    if audio_bytes:
        return _write_audio(audio_bytes, "prompt")

    try:
        detailed = music_api.compose_detailed(
            prompt=prompt,
            music_length_ms=duration_ms,
            output_format=output_format,
            force_instrumental=force_instrumental,
        )
        meta = getattr(detailed, "json", None)
        if meta:
            try:
                if isinstance(meta, str):
                    metadata_path.write_text(meta, encoding="utf-8")
                else:
                    metadata_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass
        audio_bytes = getattr(detailed, "audio", None)
        if isinstance(audio_bytes, (bytes, bytearray)) and audio_bytes:
            return _write_audio(bytes(audio_bytes), "compose_detailed")
        fallback_bytes = _maybe_extract_audio_bytes(detailed)
        if fallback_bytes:
            return _write_audio(fallback_bytes, "compose_detailed")
    except Exception as exc:
        log(f"[music] compose_detailed failed: {exc}")

    log("[music] Could not obtain audio bytes from ElevenLabs music API; skipping.")
    return None


def mix_narration_and_music(
        narration_path: str,
        music_path: str,
        output_path: str,
        *,
        music_volume: float = 0.2,
        threshold: float = 0.05,
        ratio: float = 8.0,
        attack_ms: int = 5,
        release_ms: int = 250,
        logger: Optional[callable] = None,
) -> bool:
    """Mix narration and music with sidechain ducking using ffmpeg.

    Returns True on success, False otherwise.
    """
    log = logger or (lambda msg: None)
    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", str(narration_path),
            "-i", str(music_path),
            "-filter_complex",
            (
                f"[1:a]volume={music_volume}[m];"
                f"[m][0:a]sidechaincompress=threshold={threshold}:ratio={ratio}:attack={attack_ms}:release={release_ms}[duck];"
                f"[0:a][duck]amix=inputs=2:duration=first:dropout_transition=0[out]"
            ),
            "-map", "[out]",
            "-c:a", "aac",
            "-b:a", "192k",
            "-ar", "44100",
            "-ac", "2",
            "-movflags", "+faststart",
            str(output_path),
        ]

        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, text=True)
        if proc.returncode != 0:
            log(f"[mix] ffmpeg failed: {proc.stderr.splitlines()[-1] if proc.stderr else 'unknown error'}")
            return False
        if not Path(output_path).exists() or Path(output_path).stat().st_size <= 0:
            log("[mix] output not created or empty.")
            return False
        log(f"[mix] mixed audio created -> {output_path}")
        return True
    except Exception as exc:
        log(f"[mix] exception: {exc}")
        return False
