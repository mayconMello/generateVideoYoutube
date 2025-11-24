from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Protocol

from PIL import Image

from src.core.download_to import safe_download_to
from src.executor.adapters.editly_adapter import EditlyAdapter
from src.executor.providers.base import ProviderContext
from src.executor.providers.factory import ProviderFactory
from src.media.asset_selector import AssetSelector
from src.pipeline.video_profiles import VideoProfile
from src.schemas.recipe import Asset, Policy, Scene, VideoRecipe

VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".mkv"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


class RenderAdapter(Protocol):
    def render(
        self,
        *,
        render_plan: Dict,
        audio_path: str,
        output_path: str,
        workdir: str,
    ) -> str:
        ...


@dataclass
class AssetDecision:
    scene_index: int
    asset_index: int
    source_preference: str
    acquisition: str
    path: Optional[str]
    score: Optional[float]
    note: str
    metrics: Dict[str, object] = field(default_factory=dict)


@dataclass
class RecipeExecutionResult:
    final_video_path: str
    render_plan: Dict
    selection_notes: List[AssetDecision]
    render_policy: Dict


def _ensure_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".mkv"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def _asset_dest(base_dir: str, scene_idx: int, asset_idx: int, ext: str) -> str:
    return os.path.join(base_dir, f"scene_{scene_idx:02d}_asset_{asset_idx:02d}{ext}")


def _scene_duration(scene: Scene) -> float:
    start = float(scene.start_time)
    end = float(scene.end_time)
    return max(0.1, end - start)


def _serialize_policy(policy: Policy) -> Dict:
    return json.loads(policy.model_dump_json())


class RecipeExecutor:
    """
    Execute a VideoRecipe by delegating asset acquisition to providers and rendering via adapters.
    """

    def __init__(
        self,
        *,
        workdir: str,
        selector: Optional[AssetSelector] = None,
        adapters: Optional[Dict[str, RenderAdapter]] = None,
        max_stock_candidates: int = 30,  # Increased from 8 to get more stock options
        log_func=None,
    ):
        self.workdir = workdir
        self.cache_dir = _ensure_dir(os.path.join(workdir, ".cache"))
        self.assets_dir = _ensure_dir(os.path.join(workdir, "assets"))
        self.max_stock_candidates = max_stock_candidates
        self.log = log_func or (lambda msg: None)

        self.selector = selector or AssetSelector()
        self.adapters = adapters or {
            "editly": EditlyAdapter(),
        }
        self.provider_factory = ProviderFactory(
            selector=self.selector,
            cache_dir=self.cache_dir,
            max_candidates=self.max_stock_candidates,
        )

    def execute(
        self,
        recipe: VideoRecipe,
        *,
        audio_path: str,
        alignment_path: str,
        video_profile: VideoProfile | None = None,
        segments: Optional[List[dict]] = None,
    ) -> RecipeExecutionResult:
        render_dict = json.loads(recipe.model_dump_json())
        selection_notes: List[AssetDecision] = []

        profile_dict = video_profile.to_dict() if video_profile else None
        if video_profile:
            policy_render = render_dict.get("policy", {}).get("render", {})
            policy_render["fps"] = video_profile.fps
            policy_render["profile"] = video_profile.key
            policy_render["width"] = video_profile.width
            policy_render["height"] = video_profile.height
            policy_render["aspect_ratio"] = video_profile.aspect_ratio
            policy_render["fade_in_sec"] = video_profile.fade_in_sec
            policy_render["fade_out_sec"] = video_profile.fade_out_sec
            render_dict.setdefault("policy", {})["render"] = policy_render
            render_dict["video_profile"] = profile_dict

        segment_overrides = self._prepare_segment_overrides(recipe, segments)

        for scene_idx, scene in enumerate(recipe.scenes):
            scene_entry = render_dict["scenes"][scene_idx]
            assets_out = scene_entry.get("assets", [])
            if segment_overrides:
                override_start, override_end = segment_overrides[scene_idx]
                if (
                    abs(scene.start_time - override_start) > 1e-3
                    or abs(scene.end_time - override_end) > 1e-3
                ):
                    self.log(
                        f"[executor] scene={scene.index} timing override "
                        f"{scene.start_time:.3f}-{scene.end_time:.3f} -> {override_start:.3f}-{override_end:.3f}"
                    )
                scene.start_time = override_start
                scene.end_time = override_end
                scene_entry["start_time"] = override_start
                scene_entry["end_time"] = override_end

            scene_duration = _scene_duration(scene)
            total_hint = sum(max(0.1, a.duration_hint_sec) for a in scene.assets) or scene_duration
            scale = scene_duration / total_hint if total_hint > 0 else 1.0

            for asset_idx, asset in enumerate(scene.assets):
                if asset_idx >= len(assets_out):
                    continue

                final_path, decision = self._resolve_asset(
                    recipe=recipe,
                    scene=scene,
                    asset=asset,
                    scene_idx=scene_idx,
                    asset_idx=asset_idx,
                )
                selection_notes.append(decision)

                asset_entry = assets_out[asset_idx]
                asset_entry["local_path"] = final_path
                asset_entry["selection_note"] = decision.note
                asset_entry["selection_metrics"] = decision.metrics
                asset_entry["selection_score"] = decision.score

                hint = max(0.1, asset.duration_hint_sec)
                duration = hint * scale
                asset_entry["duration_sec"] = round(duration, 3)

        adapter_key = recipe.policy.render.adapter
        adapter = self.adapters.get(adapter_key)
        if adapter is None:
            raise RuntimeError(f"Unknown render adapter '{adapter_key}'")

        output_path = os.path.join(self.workdir, "final_video.mp4")
        final_video = adapter.render(
            render_plan={
                "recipe": render_dict,
                "audio_path": audio_path,
                "alignment_path": alignment_path,
                "workdir": self.workdir,
                "video_profile": profile_dict,
            },
            audio_path=audio_path,
            output_path=output_path,
            workdir=self.workdir,
        )

        return RecipeExecutionResult(
            final_video_path=final_video,
            render_plan=render_dict,
            selection_notes=selection_notes,
            render_policy=_serialize_policy(recipe.policy),
        )

    # ------------------------------------------------------------------ internal helpers

    def _resolve_asset(
        self,
        *,
        recipe: VideoRecipe,
        scene: Scene,
        asset: Asset,
        scene_idx: int,
        asset_idx: int,
    ) -> Tuple[str, AssetDecision]:
        predefined_path = self._existing_asset_path(scene_idx, asset_idx, asset.type)
        if predefined_path:
            self.log(
                f"[executor] reuse scene={scene.index} asset={asset_idx} path={predefined_path}"
            )
            return predefined_path, AssetDecision(
                scene_index=scene_idx,
                asset_index=asset_idx,
                source_preference=asset.source_preference,
                acquisition="reuse",
                path=predefined_path,
                score=None,
                note="reused-existing",
                metrics={},
            )

        provider = self.provider_factory.get_provider(asset.source_preference or "either")
        context = ProviderContext(
            recipe=recipe,
            scene=scene,
            asset=asset,
            scene_index=scene_idx,
            asset_index=asset_idx,
            workdir=self.workdir,
            logger=self.log,
        )

        result = provider.acquire(context)
        final_path = self._finalize_asset(result.path, scene_idx, asset_idx, asset.type)

        metrics = result.metrics or {}
        metrics.setdefault("source_preference", result.source_preference)

        actual_type = _infer_media_type(final_path)
        note = result.note
        if actual_type != asset.type:
            metrics["override_type"] = actual_type
            self.log(
                f"[executor] adjusted type scene={scene.index} asset={asset_idx} {asset.type}->{actual_type}"
            )
            note = f"{result.note} (type {asset.type}->{actual_type})"

        return final_path, AssetDecision(
            scene_index=scene_idx,
            asset_index=asset_idx,
            source_preference=result.source_preference,
            acquisition=result.acquisition,
            path=final_path,
            score=result.score,
            note=note,
            metrics=metrics,
        )

    def _finalize_asset(self, source_path: str, scene_idx: int, asset_idx: int, asset_type: str) -> str:
        source_ext = Path(source_path).suffix.lower()
        if not source_ext:
            source_ext = ".mp4" if asset_type == "video" else ".jpg"
        dest = _asset_dest(self.assets_dir, scene_idx, asset_idx, source_ext)
        os.makedirs(os.path.dirname(dest), exist_ok=True)

        if os.path.abspath(source_path) == os.path.abspath(dest):
            return dest

        if os.path.exists(source_path):
            shutil.copyfile(source_path, dest)
        else:
            safe_download_to(dest, source_path)

        if asset_type == "image":
            dest = _ensure_image_format(dest)
        return dest

    def _existing_asset_path(self, scene_idx: int, asset_idx: int, asset_type: str) -> Optional[str]:
        """
        Finds any previously generated asset on disk, preferring files that match the requested type.

        Wrapped renders may downgrade a \"video\" asset to an image (or vice-versa) if the provider
        only returns one modality. When re-running we still want to reuse that asset instead of
        redownloading, so we first look for matching extensions and then fall back to any known type.
        """

        def candidate_paths(exts: List[str]) -> List[str]:
            return [_asset_dest(self.assets_dir, scene_idx, asset_idx, ext) for ext in exts]

        preferred_exts = list(VIDEO_EXTENSIONS) if asset_type == "video" else list(IMAGE_EXTENSIONS)
        fallback_exts = list(IMAGE_EXTENSIONS) if asset_type == "video" else list(VIDEO_EXTENSIONS)

        candidates = candidate_paths(preferred_exts) + candidate_paths(fallback_exts)

        for path in candidates:
            if os.path.exists(path):
                actual_type = _infer_media_type(path)
                if actual_type != asset_type:
                    self.log(
                        f"[executor] reuse scene={scene_idx} asset={asset_idx} existing_type={actual_type} "
                        f"(requested={asset_type}) path={path}"
                    )
                return path
        return None

    def _prepare_segment_overrides(
        self,
        recipe: VideoRecipe,
        segments: Optional[List[dict]],
    ) -> Optional[List[tuple[float, float]]]:
        if not segments:
            return None
        if len(segments) != len(recipe.scenes):
            self.log(
                "[executor] segment count mismatch: "
                f"{len(segments)} segments vs {len(recipe.scenes)} scenes; skipping timing overrides."
            )
            return None

        overrides: List[tuple[float, float]] = []
        last_end = 0.0
        for idx, (scene, segment) in enumerate(zip(recipe.scenes, segments)):
            start = segment.get("start")
            end = segment.get("end")
            try:
                start_val = float(start)
            except (TypeError, ValueError):
                start_val = last_end
            try:
                end_val = float(end)
            except (TypeError, ValueError):
                end_val = start_val

            if start_val < last_end:
                start_val = last_end
            if end_val <= start_val:
                end_val = start_val + 0.2

            start_val = round(start_val, 3)
            end_val = round(end_val, 3)
            overrides.append((start_val, end_val))
            last_end = end_val

        return overrides


def _infer_media_type(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext in VIDEO_EXTENSIONS:
        return "video"
    return "image"


def _ensure_image_format(path: str) -> str:
    try:
        with Image.open(path) as img:
            fmt = (img.format or "").upper()
            target_path = str(Path(path).with_suffix(".jpeg"))

            if fmt not in {"JPEG", "JPG"}:
                img.convert("RGB").save(target_path, format="JPEG", quality=95)
                if target_path != path:
                    try:
                        os.remove(path)
                    except OSError:
                        pass
                return target_path

            if path != target_path:
                try:
                    os.replace(path, target_path)
                except OSError:
                    shutil.copyfile(path, target_path)
                    try:
                        os.remove(path)
                    except OSError:
                        pass
                return target_path
    except Exception:
        pass
    return path
