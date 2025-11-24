from __future__ import annotations

import os
import json
from pathlib import Path

from src.pipeline.pipeline import BaseStep, PipelineContext
from src.pipeline.video_profiles import VideoProfile
from src.schemas.recipe import VideoRecipe


class RequestRecipeStep(BaseStep):
    def __init__(self, llm_client, *, name: str = "request_recipe") -> None:
        super().__init__(name)
        self.llm_client = llm_client

    def run(self, context: PipelineContext) -> None:
        logger = context.get("logger") or (lambda msg: None)
        base_dir = context.base_dir
        profile: VideoProfile | None = context.get("video_profile")
        recipe_path = os.path.join(base_dir, "recipe.json")
        reuse = context.reuse_existing_run and Path(recipe_path).exists()

        if reuse:
            raw_text = Path(recipe_path).read_text(encoding="utf-8")
            try:
                raw_obj = json.loads(raw_text)
                # Preserve background_music_prompt in context but remove from schema for validation
                bgm_prompt = raw_obj.get("background_music_prompt")
                if isinstance(bgm_prompt, str):
                    context.set("background_music_prompt", bgm_prompt)
                if "background_music_prompt" in raw_obj:
                    raw_obj = {k: v for k, v in raw_obj.items() if k != "background_music_prompt"}
                recipe = VideoRecipe.model_validate(raw_obj)
            except Exception:
                recipe = VideoRecipe.model_validate_json(raw_text)
            logger(f"[recipe] reusing {recipe_path}")
        else:
            narration_text = context.get("narrative_text") or ""
            segments = context.get("segments") or []
            audio_path = context.get("audio_path") or context.get("narration_path") or ""
            audio_duration = 0.0
            if segments:
                audio_duration = max((float(seg.get("end", 0.0)) for seg in segments), default=0.0)
            recipe = self.llm_client.generate_recipe(
                context.topic,
                narration_text,
                segments,
                audio_path=audio_path,
                audio_duration=audio_duration,
                video_profile=profile,
            )
            Path(recipe_path).write_text(recipe.model_dump_json(indent=2, ensure_ascii=False), encoding="utf-8")
            logger(f"[recipe] generated new recipe -> {recipe_path}")

        if profile:
            self._apply_profile_overrides(recipe, profile)
            Path(recipe_path).write_text(recipe.model_dump_json(indent=2, ensure_ascii=False), encoding="utf-8")

        if recipe.policy.render.adapter != "editly":
            logger(
                f"[recipe] overriding adapter '{recipe.policy.render.adapter}' -> 'editly'"
            )
            recipe.policy.render.adapter = "editly"
            Path(recipe_path).write_text(recipe.model_dump_json(indent=2, ensure_ascii=False), encoding="utf-8")

        context.set("recipe", recipe)
        context.set("recipe_path", recipe_path)

    @staticmethod
    def _apply_profile_overrides(recipe: VideoRecipe, profile: VideoProfile) -> None:
        try:
            recipe.policy.render.fps = profile.fps
        except Exception:
            pass

        try:
            recipe.policy.render.fade_in_sec = profile.fade_in_sec
            recipe.policy.render.fade_out_sec = profile.fade_out_sec
        except Exception:
            pass

        try:
            recipe.policy.selector_policy.thresholds.min_resolution = list(profile.asset_min_resolution)
        except Exception:
            pass

        min_res = list(profile.asset_min_resolution)
        for scene in recipe.scenes:
            for asset in scene.assets:
                try:
                    asset.rules.min_resolution = min_res
                except Exception:
                    continue
