from __future__ import annotations

import json
import os
from pathlib import Path

from src.pipeline.pipeline import BaseStep, PipelineContext
from src.pipeline.services import generate_whisper_segments


class BuildSegmentsStep(BaseStep):
    def __init__(self, *, name: str = "build_segments") -> None:
        super().__init__(name)

    def run(self, context: PipelineContext) -> None:
        logger = context.get("logger") or (lambda msg: None)
        base_dir = context.base_dir
        audio_path = context.get("audio_path")
        if not audio_path:
            raise RuntimeError("Audio path missing in context.")

        segments_path = os.path.join(base_dir, "segments.json")
        alignment_path = os.path.join(base_dir, "alignment.json")
        alignment = context.get("alignment_data")
        if alignment is None:
            if Path(alignment_path).exists():
                alignment = json.loads(Path(alignment_path).read_text(encoding="utf-8"))
            else:
                raise RuntimeError("Alignment data missing; synthesize step must run first.")

        reuse = context.reuse_existing_run and Path(segments_path).exists()

        if reuse:
            segments = json.loads(Path(segments_path).read_text(encoding="utf-8"))
            logger(f"[segments] reusing {segments_path}")
        else:
            segments = generate_whisper_segments(audio_path, logger=logger)
            Path(segments_path).write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")
            logger(f"[segments] generated {len(segments)} segments via Whisper")

        context.set("segments", segments)
        context.set("segments_path", segments_path)
        context.set("alignment_data", alignment)
        context.set("alignment_path", alignment_path)
