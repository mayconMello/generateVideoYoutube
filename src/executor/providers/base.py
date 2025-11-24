from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Protocol

from src.schemas.recipe import Asset, Scene, VideoRecipe


@dataclass
class ProviderContext:
    recipe: VideoRecipe
    scene: Scene
    asset: Asset
    scene_index: int
    asset_index: int
    workdir: str
    logger: Optional[Callable[[str], None]] = None


@dataclass
class AcquisitionResult:
    path: str
    note: str
    metrics: dict
    source_preference: str
    acquisition: str
    score: float | None = None
    decision: Optional[str] = None


class AssetProvider(Protocol):
    def acquire(self, ctx: ProviderContext) -> AcquisitionResult:
        ...
