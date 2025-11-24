from __future__ import annotations

import json
import os
from pathlib import Path

from src.pipeline.pipeline import BaseStep, PipelineContext


class PersistMetadataStep(BaseStep):
    def __init__(self, *, name: str = "persist_metadata") -> None:
        super().__init__(name)

    def run(self, context: PipelineContext) -> None:
        logger = context.get("logger") or (lambda msg: None)
        result = context.get("executor_result")
        if result is None:
            raise RuntimeError("Executor result not available in context.")

        segments = context.get("segments") or []
        selection_notes = []
        for decision in result.selection_notes:
            entry = {
                "scene_index": decision.scene_index,
                "asset_index": decision.asset_index,
                "source_preference": decision.source_preference,
                "acquisition": decision.acquisition,
                "path": decision.path,
                "score": decision.score,
                "note": decision.note,
                "metrics": decision.metrics,
            }
            selection_notes.append(entry)

        scene_timings = [
            {
                "index": scene.get("index"),
                "start_time": scene.get("start_time"),
                "end_time": scene.get("end_time"),
                "text": scene.get("text"),
                "intent": scene.get("intent"),
            }
            for scene in result.render_plan.get("scenes", [])
        ]

        metadata = {
            "topic": context.topic,
            "segments": segments,
            "render_plan": result.render_plan,
            "policy": result.render_policy,
            "selection_notes": selection_notes,
            "scene_timings": scene_timings,
            "selection_summary": _build_selection_summary(selection_notes),
        }

        # Optional background music + final audio info
        bgm_prompt = context.get("background_music_prompt")
        bgm_path = context.get("background_music_path")
        final_audio = context.get("audio_path")
        if bgm_prompt or bgm_path or final_audio:
            metadata.update(
                {
                    "background_music_prompt": bgm_prompt,
                    "background_music_path": bgm_path,
                    "final_audio_path": final_audio,
                }
            )

        captions_path = context.get("captions_path")
        if captions_path:
            metadata["captions_overlay"] = {
                "style": "tiktok_karaoke",
                "ass_path": captions_path,
            }

        metadata_path = os.path.join(context.base_dir, "metadata.json")
        Path(metadata_path).write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        context.set("metadata_path", metadata_path)
        logger(f"[metadata] persisted metadata -> {metadata_path}")


def _build_selection_summary(selection_notes: list[dict]) -> list[dict]:
    """Build per-scene decision cards summarizing selector reasoning.

    For each scene, include a list of assets with:
      - provider (when available), acquisition mode, decision note
      - queries tested, positive/negative refs
      - chosen metrics (clip_sim, ocr_ratio, sharpness_norm)
      - top-3 candidate scores
    """
    if not selection_notes:
        return []
    from collections import defaultdict

    grouped = defaultdict(list)
    for note in selection_notes:
        try:
            grouped[int(note.get("scene_index", -1))].append(note)
        except Exception:
            continue

    scene_summaries: list[dict] = []
    for scene_idx, notes in sorted(grouped.items()):
        assets_summary: list[dict] = []
        for n in sorted(notes, key=lambda x: int(x.get("asset_index", 0))):
            metrics = n.get("metrics") or {}
            chosen = (metrics.get("chosen") or {})
            candidates = metrics.get("candidates") or []
            cmeta = metrics.get("candidates_meta") or []
            # queries tested from variants
            variants = []
            try:
                variants = [m.get("variant") for m in cmeta if m.get("variant")]
            except Exception:
                variants = []
            # top-3 scores
            try:
                scores = sorted(
                    [float(c.get("score", 0.0)) for c in candidates if isinstance(c.get("score"), (int, float))],
                    reverse=True,
                )
                topk = scores[:3]
            except Exception:
                topk = []
            assets_summary.append(
                {
                    "asset_index": n.get("asset_index"),
                    "acquisition": n.get("acquisition"),
                    "provider": metrics.get("provider"),
                    "decision": n.get("note"),
                    "accepted_path": n.get("path"),
                    "score": n.get("score"),
                    "intent": (metrics.get("intent") or None),
                    "queries_tested": variants,
                    "positive_refs": (metrics.get("positive_refs") or []),
                    "negative_refs": (metrics.get("negative_refs") or []),
                    "chosen_metrics": {
                        "clip_sim": chosen.get("clip_sim"),
                        "ocr_ratio": chosen.get("ocr_ratio"),
                        "sharpness_norm": chosen.get("sharpness_norm"),
                        "faces": chosen.get("faces"),
                        "abstract_flag": chosen.get("abstract_flag"),
                    },
                    "topk_scores": topk,
                }
            )

        scene_summaries.append({"scene_index": scene_idx, "assets": assets_summary})
    return scene_summaries
