from __future__ import annotations

import os
import re
from pathlib import Path

from src.pipeline.pipeline import BaseStep, PipelineContext
from src.schemas.recipe import NarrativePlan


def _strip_audio_tags(text: str) -> str:
    without_tags = re.sub(r"\[[^\]\n]+\]", "", text)
    without_tags = re.sub(r"[ \t]{2,}", " ", without_tags)
    without_tags = re.sub(r"\n[ \t]+", "\n", without_tags)
    return without_tags.strip()


class GenerateNarrativeStep(BaseStep):
    def __init__(self, llm_client, *, name: str = "generate_narrative") -> None:
        super().__init__(name)
        self.llm_client = llm_client

    def run(self, context: PipelineContext) -> None:
        logger = context.get("logger") or (lambda msg: None)
        base_dir = context.base_dir
        narrative_path = os.path.join(base_dir, "narration.txt")
        tts_path = os.path.join(base_dir, "narration_tts.txt")

        reuse_allowed = context.reuse_existing_run and Path(narrative_path).exists()
        tts_exists = Path(tts_path).exists()

        if reuse_allowed and tts_exists:
            clean_text = Path(narrative_path).read_text(encoding="utf-8")
            tts_text = Path(tts_path).read_text(encoding="utf-8")
            logger(f"[narrative] reusing {narrative_path} (clean) and {tts_path} (tts)")
        elif reuse_allowed:
            clean_text = Path(narrative_path).read_text(encoding="utf-8")
            tts_text = clean_text
            Path(tts_path).write_text(tts_text, encoding="utf-8")
            logger(f"[narrative] reusing {narrative_path} and creating missing {tts_path}")
        else:
            plan: NarrativePlan = self.llm_client.generate_narrative(context.topic)
            tts_text = plan.narration_text.strip()
            clean_text = _strip_audio_tags(tts_text)

            Path(tts_path).write_text(tts_text, encoding="utf-8")
            Path(narrative_path).write_text(clean_text, encoding="utf-8")
            logger(f"[narrative] generated new narrative with audio tags -> {tts_path}")

        context.set("narrative_text", clean_text)
        context.set("narrative_path", narrative_path)
        context.set("narrative_text_tts", tts_text)
        context.set("narrative_tts_path", tts_path)
