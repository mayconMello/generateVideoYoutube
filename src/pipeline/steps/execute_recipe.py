from __future__ import annotations

import json
import os
from pathlib import Path

from src.executor import RecipeExecutor
# Importar o cliente para controlar o ciclo de vida
from src.media.generation.runpod_flux import flux_client
from src.pipeline.pipeline import BaseStep, PipelineContext


class ExecuteRecipeStep(BaseStep):
    def __init__(self, *, name: str = "execute_recipe") -> None:
        super().__init__(name)

    def run(self, context: PipelineContext) -> None:
        recipe = context.get("recipe")
        if recipe is None:
            raise RuntimeError("Recipe not available in context.")

        audio_path = context.get("audio_path")
        alignment_path = context.get("alignment_path")
        if not audio_path or not alignment_path:
            raise RuntimeError("Narration assets missing in context.")

        logger = context.get("logger") or (lambda msg: None)
        segments = context.get("segments")

        logger("[executor] 🚀 Iniciando Pod no RunPod (pode levar alguns segundos)...")
        try:
            flux_client.ensure_pod_active()
            logger("[executor] ✅ Pod ativo e pronto para gerar!")
        except Exception as e:
            logger(f"[executor] ⚠️ Aviso: Falha ao verificar Pod: {e}")
        try:
            logger("[executor] starting recipe execution")
            executor = RecipeExecutor(workdir=context.base_dir, log_func=logger)
            video_profile = context.get("video_profile")
            result = executor.execute(
                recipe,
                audio_path=audio_path,
                alignment_path=alignment_path,
                video_profile=video_profile,
                segments=segments,
            )

            context.set("executor_result", result)
            context.set("final_video_path", result.final_video_path)

            render_plan_path = os.path.join(context.base_dir, "render_plan.json")
            with open(render_plan_path, "w", encoding="utf-8") as fp:
                fp.write(json.dumps(result.render_plan, ensure_ascii=False, indent=2))
            context.set("render_plan_path", render_plan_path)

            recipe_path = context.get("recipe_path")
            if recipe_path:
                try:
                    bgm_prompt = context.get("background_music_prompt")
                    data = json.loads(recipe.model_dump_json(ensure_ascii=False))
                    if bgm_prompt:
                        data["background_music_prompt"] = bgm_prompt
                    Path(recipe_path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
                except Exception:
                    Path(recipe_path).write_text(recipe.model_dump_json(indent=2, ensure_ascii=False), encoding="utf-8")

            logger(f"[executor] completed execution -> {result.final_video_path}")

        finally:
            # --- PASSO 2: Desligar a "Fábrica" (Sempre executa) ---
            logger("[executor] 🛑 Encerrando Pod para economizar créditos...")
            try:
                flux_client.terminate_pod()
                logger("[executor] ✅ Pod encerrado com sucesso.")
            except Exception as e:
                logger(f"[executor] ❌ Falha ao encerrar Pod: {e}")
