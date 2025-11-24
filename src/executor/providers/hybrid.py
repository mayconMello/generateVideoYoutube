from __future__ import annotations

from src.executor.providers.base import AcquisitionResult, AssetProvider, ProviderContext
from src.executor.providers.generated import GeneratedAssetProvider
from src.executor.providers.stock import StockAssetProvider


class HybridAssetProvider(AssetProvider):
    def __init__(
        self,
        *,
        stock_provider: StockAssetProvider,
        generated_provider: GeneratedAssetProvider,
    ) -> None:
        self.stock_provider = stock_provider
        self.generated_provider = generated_provider

    def acquire(self, ctx: ProviderContext) -> AcquisitionResult:
        logger = ctx.logger or (lambda msg: None)
        try:
            stock_result = self.stock_provider.acquire(ctx)
        except Exception as exc:
            logger(f"[hybrid] stock failed for scene={ctx.scene_index} asset={ctx.asset_index}: {exc}")
            return self.generated_provider.acquire(ctx)
        decision = (stock_result.decision or "accepted").lower()
        if decision == "accepted":
            return stock_result

        logger(
            f"[hybrid] stock decision={decision} for scene={ctx.scene_index} asset={ctx.asset_index}; attempting generated fallback"
        )
        try:
            return self.generated_provider.acquire(ctx)
        except Exception as exc:
            logger(
                f"[hybrid] generated fallback failed for scene={ctx.scene_index} asset={ctx.asset_index}: {exc}; using stock result"
            )
            return stock_result
