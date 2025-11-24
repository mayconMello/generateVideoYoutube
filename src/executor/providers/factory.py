from __future__ import annotations

import os
from dataclasses import dataclass

from src.executor.providers.base import AssetProvider, ProviderContext
from src.executor.providers.generated import GeneratedAssetProvider
from src.executor.providers.hybrid import HybridAssetProvider
from src.executor.providers.stock import StockAssetProvider
from src.media.asset_selector import AssetSelector


@dataclass
class ProviderFactory:
    selector: AssetSelector
    cache_dir: str
    max_candidates: int = 30  # Increased from 8 to get more stock options (90 total per provider with 3 queries)

    def stock_provider(self, *, cache_subdir: str = "stock") -> StockAssetProvider:
        return StockAssetProvider(
            selector=self.selector,
            cache_dir=os.path.join(self.cache_dir, cache_subdir),
            max_candidates=self.max_candidates,
        )

    def generated_provider(self) -> GeneratedAssetProvider:
        return GeneratedAssetProvider(
            reference_stock_provider=self.stock_provider(cache_subdir="reference_stock"),
        )

    def hybrid_provider(self) -> HybridAssetProvider:
        return HybridAssetProvider(
            stock_provider=self.stock_provider(),
            generated_provider=self.generated_provider(),
        )

    def get_provider(self, source_preference: str) -> AssetProvider:
        pref = (source_preference or "either").lower()
        if pref == "stock":
            return self.hybrid_provider()
        if pref == "generated":
            return self.generated_provider()
        return self.hybrid_provider()
