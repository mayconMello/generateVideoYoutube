from __future__ import annotations

import os
import uuid
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import Dict, List, Optional, Sequence, Tuple

from src.core.download_to import safe_download_to
from src.executor.providers.base import AcquisitionResult, AssetProvider, ProviderContext
from src.media.compliance import is_allowed_domain
from src.media.search.google_cse import search_images as search_google
from src.media.search.pexels import search_images as search_pexels
from src.media.search.pixabay import search_images as search_pixabay
from src.media.asset_selector import AssetSelector, SelectionResult


class StockAssetProvider(AssetProvider):
    def __init__(
        self,
        *,
        selector: AssetSelector,
        cache_dir: str,
        max_candidates: int = 30,  # Increased from 8 to fetch more options from stock APIs
    ) -> None:
        self.selector = selector
        self.cache_dir = cache_dir
        # Allow environment override
        env_max = os.getenv("STOCK_MAX_CANDIDATES")
        if env_max:
            try:
                max_candidates = int(env_max)
            except ValueError:
                pass
        self.max_candidates = max(1, min(max_candidates, 50))  # Clamp between 1-50

    def acquire(self, ctx: ProviderContext) -> AcquisitionResult:
        logger = ctx.logger or (lambda msg: None)

        recipe = ctx.recipe
        scene = ctx.scene
        asset = ctx.asset
        selector_policy = recipe.policy.selector_policy
        thresholds = selector_policy.thresholds.model_dump()
        weights = selector_policy.weights.model_dump()
        limits = selector_policy.limits.model_dump()

        strategy = _normalize_strategy(scene.intent, asset.rules.search_strategy)
        deny_terms = recipe.policy.search_policy.deny_terms
        whitelist = recipe.policy.search_policy.whitelist_domains

        logger(
            f"[stock] scene={scene.index} asset={ctx.asset_index} intent={scene.intent} queries={asset.search_queries}"
        )

        relax_on_failure = asset.source_preference != "generated"
        try:
            relax_clip_margin = float(limits.get("relax_clip_margin", 0.05))
        except (TypeError, ValueError, AttributeError):
            relax_clip_margin = 0.05

        pick, candidate_meta = _run_stock_search(
            search_queries=asset.search_queries,
            strategy=strategy,
            scene=scene,
            asset=asset,
            deny_terms=deny_terms,
            whitelist=whitelist,
            thresholds=thresholds,
            weights=weights,
            limits=limits,
            selector=self.selector,
            cache_dir=self.cache_dir,
            max_candidates=self.max_candidates,
            logger=logger,
            relax_on_failure=relax_on_failure,
            relax_clip_margin=relax_clip_margin,
        )

        if not pick.path:
            raise RuntimeError(f"Stock selection failed: {pick.note}")

        metrics = pick.metrics or {}
        metrics["candidates_meta"] = candidate_meta
        logger(
            f"[stock] accepted scene={scene.index} asset={ctx.asset_index} score={pick.score} note='{pick.note}'"
        )

        # Persist JSONL audit for selector decisions
        try:
            artifacts_dir = os.path.join(ctx.workdir, "artifacts")
            os.makedirs(artifacts_dir, exist_ok=True)
            jsonl_path = os.path.join(artifacts_dir, "selector_decisions.jsonl")
            record = {
                "scene_index": ctx.scene_index,
                "asset_index": ctx.asset_index,
                "intent": scene.intent,
                "source_preference": asset.source_preference,
                "search_strategy": strategy,
                "search_queries": asset.search_queries,
                "positive_refs": (metrics.get("positive_refs") or []),
                "negative_refs": (metrics.get("negative_refs") or []),
                "thresholds": thresholds,
                "limits": limits,
                "weights": weights,
                "pick": {
                    "path": pick.path,
                    "score": pick.score,
                    "note": pick.note,
                    "decision": getattr(pick, "decision", None),
                },
                "candidates": metrics.get("candidates", []),
                "candidates_meta": candidate_meta,
            }
            with open(jsonl_path, "a", encoding="utf-8") as fp:
                import json as _json
                fp.write(_json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass

        return AcquisitionResult(
            path=pick.path,
            note=pick.note or "stock-selected",
            metrics=metrics,
            source_preference=asset.source_preference,
            acquisition="stock",
            score=pick.score,
            decision=getattr(pick, "decision", None),
        )


SUPPORTED_PROVIDERS = {"google", "pexels", "pixabay"}


def _normalize_strategy(intent: str, strategy: Sequence[str] | None) -> List[str]:
    if strategy:
        ordered: List[str] = []
        seen = set()
        for item in strategy:
            provider = (item or "").strip().lower()
            if provider in SUPPORTED_PROVIDERS and provider not in seen:
                ordered.append(provider)
                seen.add(provider)
        if ordered:
            return ordered

    default_order = ["google", "pexels", "pixabay"]
    if (intent or "").lower() != "factual":
        default_order.remove("google")
        default_order.append("google")
    return default_order


def _expand_variants(queries: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    variants: List[str] = []
    for query in queries:
        normalized = " ".join((query or "").strip().split())
        if not normalized:
            continue
        lower = normalized.lower()
        if lower in seen:
            continue
        variants.append(normalized)
        seen.add(lower)
    return variants


def _run_stock_search(
    *,
    search_queries: Sequence[str],
    strategy: Sequence[str],
    scene,
    asset,
    deny_terms: Sequence[str],
    whitelist: Sequence[str],
    thresholds: Dict,
    weights: Dict,
    limits: Dict,
    selector: AssetSelector,
    cache_dir: str,
    max_candidates: int,
    logger,
    relax_on_failure: bool,
    relax_clip_margin: float,
) -> Tuple[SelectionResult, List[Dict[str, object]]]:
    # Mantém o melhor resultado encontrado ao longo de TODA a estratégia,
    # em vez de retornar no primeiro provider que passar. Isso garante que
    # seguimos a sequência completa em search_strategy e escolhemos o melhor
    # candidato por score.
    best_pick: Optional[SelectionResult] = None
    best_meta: List[Dict[str, object]] = []

    variants = _expand_variants(search_queries)
    for provider in strategy:
        if provider not in SUPPORTED_PROVIDERS:
            logger(f"[stock] provider '{provider}' not supported; skipping")
            continue

        candidate_paths: List[str] = []
        candidate_meta: List[Dict[str, object]] = []

        for variant in variants:
            urls = _search_provider(
                provider=provider,
                query=variant,
                deny_terms=deny_terms,
                whitelist=whitelist,
                limit=max_candidates,
            )
            logger(
                f"[stock] provider={provider} variant='{variant}' results={len(urls)}"
            )
            for url in urls:
                candidate_meta.append(
                    {
                        "url": url,
                        "provider": provider,
                        "variant": variant,
                        "whitelisted": is_allowed_domain(url, whitelist),
                    }
                )
            downloaded = _download_candidates(
                cache_dir,
                urls,
                limit=max_candidates,
            )
            candidate_paths.extend(downloaded)

        if not candidate_paths:
            logger(f"[stock] provider={provider} produced no candidates")
            continue

        logger(
            f"[stock] provider={provider} evaluating {len(candidate_paths)} candidates"
        )

        # Construir visual_query rico usando semantic_text + variants para validação CLIP
        # search_queries são usadas apenas para buscar nas APIs, não para validação semântica
        semantic_parts = []
        if asset.semantic_text and isinstance(asset.semantic_text, str):
            semantic_parts.append(asset.semantic_text.strip())
        if asset.semantic_text_variants and isinstance(asset.semantic_text_variants, list):
            for variant in asset.semantic_text_variants[:2]:  # Primeiras 2 variants
                if variant and isinstance(variant, str):
                    semantic_parts.append(variant.strip())

        # Fallback para search_queries se semantic_text não existir ou estiver vazio
        visual_query_rich = " | ".join(semantic_parts) if semantic_parts else " | ".join(search_queries[:3] if search_queries else [])

        pick = selector.select_images(
            scene_text=scene.text,
            visual_query=visual_query_rich,  # ✅ Usando semantic_text rico para validação CLIP!
            paths=candidate_paths,
            scene_intent=scene.intent,
            clip_threshold=asset.rules.clip_threshold,
            overlay_limit=asset.rules.overlay_limit,
            weights=weights,
            thresholds=thresholds,
            limits=limits,
            relax_on_failure=relax_on_failure,
            relax_clip_margin=relax_clip_margin,
            target_duration_sec=float(getattr(asset, "duration_hint_sec", 0.0) or 0.0),
            # semantic guidance
            semantic_text=asset.semantic_text,
            semantic_text_variants=asset.semantic_text_variants,
            negative_semantic_texts=asset.negative_semantic_texts,
            grounded_entities=asset.grounded_entities,
        )

        # annotate provider for decision card
        try:
            if pick.metrics is not None:
                pick.metrics.setdefault("provider", provider)
        except Exception:
            pass

        logger(f"[stock] pick decision note='{pick.note}' score={pick.score}")

        # Em vez de retornar imediatamente ao encontrar um caminho válido,
        # seguimos para os próximos providers e mantemos o melhor por score.
        # Preferimos sempre um pick com path (aceito) ao invés de um sem path.
        if best_pick is None:
            best_pick = pick
            best_meta = candidate_meta
        else:
            best_has_path = bool(best_pick.path)
            curr_has_path = bool(pick.path)
            if (curr_has_path and not best_has_path) or (
                curr_has_path == best_has_path and (pick.score or 0.0) > (best_pick.score or 0.0)
            ):
                best_pick = pick
                best_meta = candidate_meta

    if best_pick:
        return best_pick, best_meta

    logger("[stock] no candidates gathered from strategy")
    return SelectionResult(path=None, score=0.0, note="no candidates", metrics={}, decision="empty"), []


def _search_provider(
    *,
    provider: str,
    query: str,
    deny_terms: Sequence[str],
    whitelist: Sequence[str],
    limit: int,
) -> List[str]:
    if not query:
        return []
    provider = provider.lower()
    if provider == "google":
        return search_google(
            query,
            domains_whitelist=whitelist,
            deny_terms=deny_terms,
            num=limit,
        )
    if provider == "pixabay":
        return search_pixabay(
            query,
            deny_terms=deny_terms,
            per_page=limit,
        )
    if provider == "pexels":
        return search_pexels(
            query,
            per_page=limit,
        )
    return []


def _download_candidates(
    cache_dir: str,
    urls: Sequence[str],
    *,
    limit: int,
) -> List[str]:
    local_paths: List[str] = []
    os.makedirs(cache_dir, exist_ok=True)
    sanitized_urls = [u for u in urls if u]
    if limit <= 0 or not sanitized_urls:
        return local_paths

    def _attempt_download(url: str) -> Optional[str]:
        try:
            ext = os.path.splitext(url)[1].lower()
            if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
                ext = ".jpg"
            token = uuid.uuid4().hex
            dest = os.path.join(cache_dir, f"stock_{token}{ext}")
            safe_download_to(dest, url)
            return dest
        except Exception:
            return None

    max_workers_env = os.getenv("STOCK_DOWNLOAD_WORKERS")
    try:
        configured_workers = int(max_workers_env) if max_workers_env else 0
    except ValueError:
        configured_workers = 0

    max_workers = configured_workers or 4
    max_workers = max(1, min(max_workers, limit, len(sanitized_urls)))

    if max_workers == 1:
        for url in sanitized_urls:
            if len(local_paths) >= limit:
                break
            result = _attempt_download(url)
            if result:
                local_paths.append(result)
        return local_paths

    executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="stock_dl")
    futures: Dict[Future[Optional[str]], str] = {}
    url_iter = iter(sanitized_urls)

    def _submit_next() -> None:
        if len(local_paths) >= limit:
            return
        try:
            url = next(url_iter)
        except StopIteration:
            return
        futures[executor.submit(_attempt_download, url)] = url

    try:
        for _ in range(max_workers):
            _submit_next()

        while futures and len(local_paths) < limit:
            done, _ = wait(set(futures.keys()), return_when=FIRST_COMPLETED)
            for future in done:
                url = futures.pop(future, "")
                try:
                    path = future.result()
                except Exception:
                    path = None
                if path:
                    local_paths.append(path)
                    if len(local_paths) >= limit:
                        break
                _submit_next()
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    return local_paths
