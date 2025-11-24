from __future__ import annotations

import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import List

from src.export.tiktok_captions import (
    build_caption_lines,
    build_word_tokens,
    write_ass,
)
from src.pipeline.pipeline import BaseStep, PipelineContext
from src.schemas.recipe import VideoRecipe


class BurnInCaptionsStep(BaseStep):
    def __init__(self, *, name: str = "burn_in_captions") -> None:
        super().__init__(name)

    def run(self, context: PipelineContext) -> None:
        logger = context.get("logger") or (lambda msg: None)
        recipe: VideoRecipe | None = context.get("recipe")
        alignment = context.get("alignment_data")
        final_video = context.get("final_video_path")

        if not recipe or not alignment or not final_video:
            logger("[captions] missing recipe/alignment/final video; skipping burn-in.")
            return

        tokens = build_word_tokens(alignment)
        if not tokens:
            logger("[captions] no tokens derived from alignment; skipping.")
            return

        captions = build_caption_lines(recipe, tokens)
        if not captions:
            logger("[captions] no captions aligned to scenes; skipping.")
            return

        ass_path = Path(context.base_dir).joinpath("tiktok_captions.ass")
        write_ass(captions, ass_path)
        logger(f"[captions] ASS overlay ready -> {ass_path}")

        video_path = Path(final_video)
        if not self._wait_for_video_ready(video_path, logger=logger):
            logger("[captions] final video not ready for burn-in; skipping.")
            return
        success = self._burn(
            video_path=video_path,
            ass_path=ass_path,
            workdir=Path(context.base_dir),
            logger=logger,
        )
        if not success:
            logger("[captions] burn-in failed; retaining original video.")
            return

        context.set("captions_path", str(ass_path))
        result = context.get("executor_result")
        if result is not None:
            result.final_video_path = str(video_path)
        logger(f"[captions] burn-in complete -> {video_path.name}")

    def _burn(
        self,
        *,
        video_path: Path,
        ass_path: Path,
        workdir: Path,
        logger,
    ) -> bool:
        tmp_out = video_path.with_name(f"{video_path.stem}_tiktok{video_path.suffix}")
        if tmp_out.exists():
            tmp_out.unlink()

        filter_arg = self._ass_filter_arg(ass_path.resolve())
        cmd: List[str] = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            filter_arg,
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "copy",
            str(tmp_out),
        ]

        logger(f"[captions] ffmpeg {' '.join(shlex.quote(part) for part in cmd)}")
        for attempt in range(2):
            proc = subprocess.run(
                cmd,
                cwd=str(workdir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if proc.returncode == 0:
                break
            logger(
                "[captions] ffmpeg failed "
                f"(code={proc.returncode}) stderr={proc.stderr.strip().splitlines()[-1] if proc.stderr else ''}"
            )
            if tmp_out.exists():
                tmp_out.unlink()
            if attempt == 0:
                time.sleep(0.75)
                continue
            return False

        try:
            os.replace(tmp_out, video_path)
        except Exception as exc:
            logger(f"[captions] could not replace original video: {exc!r}")
            if tmp_out.exists():
                tmp_out.unlink()
            return False
        return True

    @staticmethod
    def _ass_filter_arg(path: Path) -> str:
        text = str(path)
        text = text.replace("\\", "\\\\").replace(":", "\\:")
        return f"ass={text}"

    def _wait_for_video_ready(self, video_path: Path, *, logger, retries: int = 3, delay: float = 0.75) -> bool:
        for attempt in range(retries):
            if video_path.exists() and video_path.stat().st_size > 0 and self._probe_duration(video_path) is not None:
                return True
            if attempt < retries - 1:
                time.sleep(delay)
        logger("[captions] video file not readable; ffprobe could not parse duration.")
        return False

    @staticmethod
    def _probe_duration(video_path: Path) -> float | None:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode not in (0, 1):
            return None
        try:
            value = float((proc.stdout or "").strip().splitlines()[-1])
        except (ValueError, IndexError):
            return None
        return value if value > 0 else None
