from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from src.pipeline.pipeline import BaseStep, PipelineContext
from src.pipeline.services import compose_background_music, mix_narration_and_music, _probe_audio_duration


class ComposeBackgroundMusicStep(BaseStep):
    def __init__(self, eleven_client, llm_client, *, name: str = "compose_background_music") -> None:
        super().__init__(name)
        self.eleven_client = eleven_client
        self.llm_client = llm_client

    def run(self, context: PipelineContext) -> None:
        logger = context.get("logger") or (lambda msg: None)
        music_enabled = os.getenv("BACKGROUND_MUSIC_ENABLED", "true").strip().lower()
        if music_enabled not in {"1", "true", "yes", "on"}:
            logger("[music] background music disabled via BACKGROUND_MUSIC_ENABLED; skipping.")
            return

        base_dir = context.base_dir
        narration_path = context.get("audio_path")
        if not narration_path:
            raise RuntimeError("Narration audio path not found in context.")

        duration = _probe_audio_duration(narration_path)
        if not duration or duration <= 0:
            segments = context.get("segments") or []
            try:
                duration = max((float(s.get("end", 0.0)) for s in segments), default=0.0)
            except Exception:
                duration = 0.0
        if not duration or duration <= 0:
            logger("[music] could not determine narration duration; skipping background music.")
            return

        topic = context.topic
        narration_text = context.get("narrative_text") or context.get("narrative_text_tts") or ""
        prompt = context.get("background_music_prompt")
        if not prompt:
            try:
                plan = self.llm_client.generate_music_prompt(topic, narration_text, float(duration))
                prompt = plan.prompt
            except Exception as exc:
                logger(f"[music] prompt generation failed: {exc}")
                return

        try:
            Path(os.path.join(base_dir, "background_music_prompt.txt")).write_text(prompt, encoding="utf-8")
        except Exception:
            pass

        music_path: Optional[str] = compose_background_music(
            self.eleven_client,
            prompt=prompt,
            duration_sec=float(duration),
            base_dir=base_dir,
            logger=logger,
            force_instrumental=True,
        )
        if not music_path:
            logger("[music] composition unavailable; keeping narration-only audio.")
            return

        mixed_path = os.path.join(base_dir, "narration_with_bgm.m4a")
        ok = mix_narration_and_music(narration_path, music_path, mixed_path, logger=logger)
        if not ok:
            logger("[music] mixing failed; keeping narration-only audio.")
            return

        context.set("background_music_path", music_path)
        context.set("background_music_prompt", prompt)
        context.set("audio_path", mixed_path)
        logger("[music] background music mixed into final audio.")
