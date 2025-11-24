from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from src.schemas.recipe import Scene, VideoRecipe


@dataclass
class WordToken:
    text: str
    start: float
    end: float


@dataclass
class CaptionLine:
    start: float
    end: float
    words: List[WordToken]
    text: str


_BREAK_RE = re.compile(r"\s+")
_PUNCT_CHARS = set(",.!?;:…\"“”()[]{}")
_HYPHENS = {"-", "–", "—"}


def build_word_tokens(alignment_data: dict) -> List[WordToken]:
    if not alignment_data:
        return []
    source = (alignment_data.get("source") or "").lower()
    if source == "whisperx" or alignment_data.get("segments"):
        return _build_tokens_from_whisperx(alignment_data)
    return _build_tokens_from_character_alignment(alignment_data)


def build_caption_lines(
    recipe: VideoRecipe,
    tokens: Sequence[WordToken],
    *,
    tolerance: float = 0.18,
) -> List[CaptionLine]:
    captions: List[CaptionLine] = []
    if not recipe.scenes:
        return captions

    token_index = 0
    total_tokens = len(tokens)

    for scene in recipe.scenes:
        scene_start = float(scene.start_time or 0.0)
        scene_end = float(scene.end_time or (scene_start + 0.5))
        scene_words: List[WordToken] = []

        while token_index < total_tokens and tokens[token_index].end < scene_start - tolerance:
            token_index += 1

        probe = token_index
        while probe < total_tokens and tokens[probe].start <= scene_end + tolerance:
            scene_words.append(tokens[probe])
            probe += 1

        if not scene_words:
            synthetic = _synthetic_tokens(scene, scene_start, scene_end)
            if synthetic:
                captions.append(
                    CaptionLine(
                        start=synthetic[0].start,
                        end=synthetic[-1].end,
                        words=synthetic,
                        text=scene.text,
                    )
                )
            continue

        captions.append(
            CaptionLine(
                start=scene_words[0].start,
                end=scene_words[-1].end,
                words=scene_words,
                text=scene.text,
            )
        )
        token_index = probe

    return captions


def _synthetic_tokens(scene: Scene, start: float, end: float) -> List[WordToken]:
    return _synthetic_word_tokens(scene.text, start, end)


def write_ass(captions: Sequence[CaptionLine], path: str | Path) -> None:
    path = Path(path)
    lines: List[str] = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "PlayResX: 1920",
        "PlayResY: 1080",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        (
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
            "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
            "MarginR, MarginV, Encoding"
        ),
        (
            "Style: TikTok,Montserrat Black,96,&H0000F8FF,&H0066FFFF,&H00000000,&H64000000,-1,0,0,0,100,100,2,0,1,10,0,2,90,90,140,1"
        ),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    for caption in captions:
        start_ts = _format_ass_timestamp(caption.start)
        end_ts = _format_ass_timestamp(caption.end)
        text = _build_dialogue_text(caption.words)
        line = f"Dialogue: 0,{start_ts},{end_ts},TikTok,,0,0,0,,{{\\fad(60,80)\\blur0.6}}{text}"
        lines.append(line)

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_tokens_from_whisperx(alignment_data: dict) -> List[WordToken]:
    tokens: List[WordToken] = []
    for segment in alignment_data.get("segments") or []:
        words = segment.get("words") or []
        if words:
            for word in words:
                text = (word.get("word") or "").strip()
                start = _coerce_time(word.get("start"))
                end = _coerce_time(word.get("end"))
                if not text or start is None or end is None:
                    continue
                tokens.append(WordToken(text=text, start=float(start), end=float(end)))
        else:
            start = _coerce_time(segment.get("start"))
            end = _coerce_time(segment.get("end"))
            if start is None or end is None:
                continue
            tokens.extend(_synthetic_word_tokens(segment.get("text"), float(start), float(end)))
    return tokens


def _build_tokens_from_character_alignment(alignment_data: dict) -> List[WordToken]:
    chars: List[str] = alignment_data.get("characters") or []
    starts: List[Optional[float]] = alignment_data.get("start_times") or []
    ends: List[Optional[float]] = alignment_data.get("end_times") or []
    tokens: List[WordToken] = []

    buffer: List[str] = []
    current_start: Optional[float] = None
    last_end: Optional[float] = None

    def flush() -> None:
        nonlocal buffer, current_start, last_end
        if not buffer or current_start is None or last_end is None:
            buffer = []
            current_start = None
            last_end = None
            return
        text = "".join(buffer).strip()
        if text:
            tokens.append(WordToken(text=text, start=float(current_start), end=float(last_end)))
        buffer = []
        current_start = None
        last_end = None

    total = len(chars)
    for idx in range(total):
        ch = chars[idx]
        ch_start = starts[idx] if idx < len(starts) else None
        ch_end = ends[idx] if idx < len(ends) else None

        if ch in {"[", "]"}:
            flush()
            continue

        if _BREAK_RE.match(ch):
            flush()
            continue

        if ch in _PUNCT_CHARS:
            flush()
            continue

        if ch in _HYPHENS and not buffer:
            continue

        timestamp = None
        if ch_start is not None:
            timestamp = float(ch_start)
        elif ch_end is not None:
            timestamp = float(ch_end)

        if timestamp is None:
            continue

        if current_start is None:
            current_start = timestamp
        if ch_end is not None:
            last_end = float(ch_end)
        else:
            last_end = timestamp

        buffer.append(ch)

    flush()
    return tokens


def _synthetic_word_tokens(text: Optional[str], start: float, end: float) -> List[WordToken]:
    text = (text or "").strip()
    if not text:
        return []
    words = [w for w in _BREAK_RE.split(text) if w]
    if not words:
        return []

    duration = max(0.2, end - start)
    slice_len = duration / len(words)
    tokens: List[WordToken] = []
    cursor = start
    for word in words:
        tokens.append(WordToken(text=word, start=cursor, end=cursor + slice_len))
        cursor += slice_len
    return tokens


def _coerce_time(value: Optional[object]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_dialogue_text(words: Sequence[WordToken], wrap_chars: int = 22) -> str:
    parts: List[str] = []
    current_line_chars = 0

    for idx, word in enumerate(words):
        duration = max(1, int(round((word.end - word.start) * 100)))
        safe_text = word.text.replace("{", r"\{").replace("}", r"\}")

        if current_line_chars and current_line_chars + len(safe_text) > wrap_chars:
            parts.append(r"\N")
            current_line_chars = 0

        parts.append(f"{{\\k{duration}}}{safe_text}")

        if idx < len(words) - 1:
            parts.append(" ")
            current_line_chars += len(safe_text) + 1
        else:
            current_line_chars += len(safe_text)

    return "".join(parts).strip()


def _format_ass_timestamp(value: float) -> str:
    value = max(0.0, float(value))
    hours = int(value // 3600)
    minutes = int((value % 3600) // 60)
    seconds = int(value % 60)
    centiseconds = int(round((value - int(value)) * 100))
    return f"{hours}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"
