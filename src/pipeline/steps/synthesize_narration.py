from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional, Union

from src.pipeline.pipeline import BaseStep, PipelineContext
from src.pipeline.services import NarrationAssets, synthesize_narration


class SynthesizeNarrationStep(BaseStep):
    def __init__(
        self,
        eleven_client,
        voice_id: str,
        *,
        voice_settings: Optional[Dict[str, Union[float, bool]]] = None,
        name: str = "synthesize_narration",
    ) -> None:
        super().__init__(name)
        self.eleven_client = eleven_client
        self.voice_id = voice_id
        self.voice_settings = voice_settings or {}

    def run(self, context: PipelineContext) -> None:
        logger = context.get("logger") or (lambda msg: None)
        base_dir = context.base_dir
        reuse = context.reuse_existing_run
        audio_path = os.path.join(base_dir, "narration_full.mp3")
        alignment_path = os.path.join(base_dir, "alignment.json")

        if reuse and Path(audio_path).exists() and Path(alignment_path).exists():
            alignment_data = json.loads(Path(alignment_path).read_text(encoding="utf-8"))
            assets = NarrationAssets(audio_path=audio_path, alignment=alignment_data)
            logger("[narration] reusing synthesized audio/alignment")
        else:
            text = context.get("narrative_text_tts") or context.get("narrative_text")
            if not text:
                raise RuntimeError("Narrative text not found in pipeline context.")
            assets = synthesize_narration(
                self.eleven_client,
                text=text,
                base_dir=base_dir,
                voice_id=self.voice_id,
                voice_settings=self.voice_settings,
                logger=logger,
            )
            Path(alignment_path).write_text(json.dumps(assets.alignment, ensure_ascii=False, indent=2), encoding="utf-8")
            logger("[narration] synthesized new audio/alignment")

        context.set("audio_path", assets.audio_path)
        context.set("alignment_data", assets.alignment)
        context.set("alignment_path", alignment_path)
