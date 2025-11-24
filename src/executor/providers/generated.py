from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image

from src.executor.providers.base import AcquisitionResult, AssetProvider, ProviderContext
from src.executor.providers.stock import StockAssetProvider
from src.media.generation import generate_image_with_qwen, generate_video_with_seedance
from src.media.generation.utils import GenerationSpec, resolve_generation_spec, safe_json
from src.schemas.recipe import Asset


class GeneratedAssetProvider(AssetProvider):
    """
    Asset provider responsible for invoking generative models (Qwen/Seedance).
    When a video needs to be generated, it will attempt to source a keyframe image
    using the same stock search policies before falling back to image generation.
    """

    def __init__(self, *, reference_stock_provider: Optional[StockAssetProvider] = None) -> None:
        self.reference_stock_provider = reference_stock_provider

    def acquire(self, ctx: ProviderContext) -> AcquisitionResult:
        logger = ctx.logger or (lambda msg: None)

        scene = ctx.scene
        asset = ctx.asset
        spec = resolve_generation_spec()

        image_prompt = asset.generate_prompt or asset.primary_search_query or scene.text
        video_prompt = asset.video_generate_prompt or image_prompt

        if asset.type == "video":
            keyframe_path, reference_info = self._prepare_reference_frame(
                ctx,
                prompt=image_prompt,
                spec=spec,
            )

            video_url, video_debug = generate_video_with_seedance(
                keyframe_path,
                prompt=video_prompt,
                duration=max(1.0, asset.duration_hint_sec or 1.0),
                fps=spec.seedance_fps,
                resolution=spec.resolution,
                aspect_ratio=spec.aspect_ratio,
            )
            if not video_url:
                raise RuntimeError(
                    f"Seedance generation failed; details={safe_json(video_debug)}"
                )

            logger(
                f"[generated] scene={scene.index} asset={ctx.asset_index} seedance url={video_url}"
            )
            video_path = _download_to_assets(
                video_url,
                ctx.workdir,
                ctx.scene_index,
                ctx.asset_index,
                suffix=".mp4",
            )

            metrics = {
                "reference_frame": reference_info,
                "video_generation": video_debug,
                "seedance_url": video_url,
            }
            return AcquisitionResult(
                path=video_path,
                note="generated-video",
                metrics=metrics,
                source_preference="generated",
                acquisition="generated",
                decision="generated",
            )

        image_url, image_debug = generate_image_with_qwen(
            image_prompt,
            aspect_ratio=spec.aspect_ratio,
            size=spec.image_size,
            width=spec.width,
            height=spec.height,
        )
        if not image_url:
            raise RuntimeError(
                f"Image generation failed; details={safe_json(image_debug)}"
            )
        logger(
            f"[generated] scene={scene.index} asset={ctx.asset_index} image url={image_url}"
        )
        image_path = _download_to_assets(
            image_url,
            ctx.workdir,
            ctx.scene_index,
            ctx.asset_index,
            suffix=".jpeg",
        )
        metrics = {"image_generation": image_debug, "image_url": image_url}
        return AcquisitionResult(
            path=image_path,
            note="generated-image",
            metrics=metrics,
            source_preference="generated",
            acquisition="generated",
            decision="generated",
        )

    # ------------------------------------------------------------------ helpers

    def _prepare_reference_frame(
        self,
        ctx: ProviderContext,
        *,
        spec: GenerationSpec,
        prompt: str,
    ) -> Tuple[str, dict]:
        logger = ctx.logger or (lambda msg: None)
        asset = ctx.asset

        if self.reference_stock_provider and asset.source_preference in {"stock", "either"}:
            logger(
                f"[generated] scene={ctx.scene.index} asset={ctx.asset_index} attempting stock keyframe"
            )
            stock_asset = _as_image_asset(asset)
            image_ctx = ProviderContext(
                recipe=ctx.recipe,
                scene=ctx.scene,
                asset=stock_asset,
                scene_index=ctx.scene_index,
                asset_index=ctx.asset_index,
                workdir=ctx.workdir,
                logger=logger,
            )
            try:
                stock_result = self.reference_stock_provider.acquire(image_ctx)
                prepared = _copy_keyframe_to_assets(
                    stock_result.path,
                    ctx.workdir,
                    ctx.scene_index,
                    ctx.asset_index,
                )
                logger(
                    f"[generated] scene={ctx.scene.index} asset={ctx.asset_index} using stock keyframe"
                )
                return prepared, {
                    "mode": "stock",
                    "note": stock_result.note,
                    "score": stock_result.score,
                    "metrics": stock_result.metrics,
                    "source_path": stock_result.path,
                }
            except Exception as exc:
                logger(
                    f"[generated] scene={ctx.scene.index} asset={ctx.asset_index} stock keyframe failed: {exc}"
                )

        logger(
            f"[generated] scene={ctx.scene.index} asset={ctx.asset_index} generating keyframe"
        )
        keyframe_url, keyframe_debug = generate_image_with_qwen(
            prompt,
            aspect_ratio=spec.aspect_ratio,
            size=spec.image_size,
            width=spec.width,
            height=spec.height,
        )
        if not keyframe_url:
            raise RuntimeError(
                f"Seedance keyframe generation failed; details={safe_json(keyframe_debug)}"
            )

        keyframe_path = _download_to_assets(
            keyframe_url,
            ctx.workdir,
            ctx.scene_index,
            ctx.asset_index,
            suffix=".jpeg",
        )
        return keyframe_path, {
            "mode": "generated",
            "image_generation": keyframe_debug,
            "image_url": keyframe_url,
        }


def _as_image_asset(asset: Asset) -> Asset:
    return asset.model_copy(
        update={
            "type": "image",
            "source_preference": "stock",
        }
    )


def _copy_keyframe_to_assets(
    source_path: str,
    workdir: str,
    scene_idx: int,
    asset_idx: int,
) -> str:
    assets_dir = os.path.join(workdir, "assets")
    os.makedirs(assets_dir, exist_ok=True)
    dest = Path(assets_dir) / f"scene_{scene_idx:02d}_asset_{asset_idx:02d}_keyframe.jpeg"
    try:
        with Image.open(source_path) as img:
            img.convert("RGB").save(dest, format="JPEG", quality=95)
    except Exception as exc:
        raise RuntimeError(f"Failed to prepare stock keyframe: {exc}") from exc
    return str(dest)


def _download_to_assets(url: str, workdir: str, scene_idx: int, asset_idx: int, suffix: str) -> str:
    from src.core.download_to import safe_download_to

    assets_dir = os.path.join(workdir, "assets")
    os.makedirs(assets_dir, exist_ok=True)
    dest = os.path.join(assets_dir, f"scene_{scene_idx:02d}_asset_{asset_idx:02d}{suffix}")
    safe_download_to(dest, url)
    return dest
