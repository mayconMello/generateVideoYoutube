from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

try:
    import pytesseract  # type: ignore
except Exception:  # pragma: no cover
    pytesseract = None

from src.pipeline.asset_scoring import (
    AssetCandidate,
    AssetScoringContext,
    AssetScoringModels,
    AssetScoreResult,
    passes_thresholds,
    score_candidate,
)


@dataclass
class SelectionResult:
    path: Optional[str]
    score: float
    note: str
    decision: str
    metrics: Optional[dict] = None


class AssetSelector:
    """Unified selector for media assets (images first).

    Implements a robust, stock-first ranking with:
    - CLIP semantic scoring (positives only; negatives penalized separately)
    - visual quality gates (resolution, sharpness)
    - overlay/ocr penalties, abstract-pattern penalty, face bonus
    - small relax margin if nothing passes strict gates

    This selector is the sole decision point; keep other pickers only as utilities.
    """

    def __init__(
        self,
        *,
        min_width: int = 1280,
        min_height: int = 720,
        default_clip_threshold_img: float = 0.27,
        default_overlay_limit: float = 0.10,
        max_ocr_chars: int = 12,
        min_sharpness: float = 0.18,
        edge_density_max: float = 0.16,
        resolution_relax_ratio: float = 0.72,
    ) -> None:
        self.min_width = int(min_width)
        self.min_height = int(min_height)
        self.default_clip_threshold_img = float(default_clip_threshold_img)
        self.default_overlay_limit = float(default_overlay_limit)
        self.max_ocr_chars = int(max_ocr_chars)
        self.min_sharpness = float(min_sharpness)
        self.edge_density_max = float(edge_density_max)
        env_relax = os.getenv("ASSET_RESOLUTION_RELAX_RATIO")
        try:
            relax_val = float(env_relax) if env_relax else float(resolution_relax_ratio)
        except (TypeError, ValueError):
            relax_val = float(resolution_relax_ratio)
        self.resolution_relax_ratio = min(1.0, max(0.4, relax_val))
        self.scoring_models = AssetScoringModels()

    # --------------------------- helpers (local, no external deps) ---------------------------
    @staticmethod
    def _read_size(path: str) -> Tuple[int, int]:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            return im.size

    @staticmethod
    def _ocr_char_count(path: str) -> int:
        if pytesseract is None:
            return 0
        try:
            with Image.open(path) as im:
                im = im.convert("L")
                w, h = im.size
                scale = 1024 / max(w, h)
                if scale < 1.0:
                    im = im.resize((int(w * scale), int(h * scale)))
                data = pytesseract.image_to_data(im, output_type="dict")
            texts = data.get("text") or []
            confs = data.get("conf") or []
            count = 0
            for t, c in zip(texts, confs):
                try:
                    c = float(c)
                except Exception:
                    c = -1.0
                t = (t or "").strip()
                if c >= 60 and len(t) >= 2 and any(ch.isalpha() for ch in t):
                    count += len(t)
            return count
        except Exception:
            return 0

    def _sharpness_metrics(self, path: str) -> Tuple[float, float]:
        if cv2 is None:
            return 0.0, 0.0
        try:
            data = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return 0.0, 0.0
            lap = cv2.Laplacian(img, cv2.CV_64F)
            variance = float(lap.var())
            norm = min(1.0, variance / 1200.0)
            return norm, variance
        except Exception:
            return 0.0, 0.0

    def _detect_faces(self, path: str) -> int:
        if cv2 is None:
            return 0
        try:
            cascade_dir = getattr(cv2, "data", None)
            if cascade_dir is None:
                return 0
            cascade_path = os.path.join(cascade_dir.haarcascades, "haarcascade_frontalface_default.xml")
            if not os.path.exists(cascade_path):
                return 0
            detector = cv2.CascadeClassifier(cascade_path)
            if detector.empty():
                return 0
            data = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is None:
                return 0
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
            if faces is None:
                return 0
            return int(len(faces))
        except Exception:
            return 0

    def _is_vector_pattern(self, path: str) -> bool:
        if cv2 is None:
            return False
        try:
            img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                return False
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 80, 160)
            edge_ratio = float((edges > 0).sum()) / float(edges.size)

            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            sat = hsv[..., 1].astype(np.float32) / 255.0
            mean_sat = float(np.mean(sat))

            return (edge_ratio >= self.edge_density_max and mean_sat <= 0.25)
        except Exception:
            return False

    # -------------------------------------------- API --------------------------------------------
    def select_images(
        self,
        *,
        scene_text: str,
        visual_query: str,
        paths: List[str],
        scene_intent: Optional[str] = None,
        clip_threshold: Optional[float] = None,
        overlay_limit: Optional[float] = None,
        thresholds: Optional[dict] = None,
        weights: Optional[dict] = None,
        limits: Optional[dict] = None,
        relax_on_failure: bool = False,
        relax_clip_margin: float = 0.02,
        target_duration_sec: Optional[float] = None,
        # semantics
        semantic_text: Optional[str] = None,
        semantic_text_variants: Optional[List[str]] = None,
        negative_semantic_texts: Optional[List[str]] = None,
        grounded_entities: Optional[List[str]] = None,
    ) -> SelectionResult:
        if not paths:
            return SelectionResult(path=None, score=0.0, note="no candidates", decision="empty", metrics={"candidates": []})

        intent = (scene_intent or "factual").strip().lower()
        if intent not in {"factual", "conceptual"}:
            intent = "factual"

        thresholds = thresholds or {}
        weights_cfg = (weights or {}).get(intent, {})
        limits = limits or {}

        def _w(name: str, default: float) -> float:
            try:
                return float(weights_cfg.get(name, default))
            except (TypeError, ValueError):
                return float(default)

        weights_resolved = {
            "clip_sim": _w("clip_sim", 1.0),
            "resolution_score": _w("resolution_score", 0.30),
            "sharpness": _w("sharpness", 0.20),
            "ocr_penalty": _w("ocr_penalty", 0.80),
            "face_bonus": _w("face_bonus", 0.15),
            "abstract_penalty": _w("abstract_penalty", 0.40),
        }

        # Gates
        min_res_cfg = thresholds.get("min_resolution") or [self.min_width, self.min_height]
        try:
            min_w = int(min_res_cfg[0])
            min_h = int(min_res_cfg[1])
        except Exception:
            min_w, min_h = self.min_width, self.min_height

        if intent == "factual":
            policy_gate = float(thresholds.get("min_clip_sim_factual", self.default_clip_threshold_img))
            env_gate = float(os.getenv("CLIP_FACTUAL_GATE_IMG", str(policy_gate)))
            min_clip_default = max(policy_gate, env_gate)
        else:
            policy_gate = float(thresholds.get("min_clip_sim_conceptual", max(0.18, self.default_clip_threshold_img - 0.03)))
            env_gate = float(os.getenv("CLIP_CONCEPTUAL_GATE_IMG", str(policy_gate)))
            min_clip_default = max(policy_gate, env_gate)
        min_clip_gate = float(clip_threshold) if clip_threshold is not None else float(min_clip_default)
        relax_env = float(os.getenv("CLIP_RELAX_MARGIN", "0.02"))

        if overlay_limit is not None:
            overlay_gate = float(overlay_limit)
        elif "overlay_limit" in thresholds:
            overlay_gate = float(thresholds.get("overlay_limit"))
        else:
            overlay_gate = self.default_overlay_limit

        max_ocr_chars = int(limits.get("max_ocr_chars", self.max_ocr_chars))
        min_sharpness_gate = float(limits.get("min_sharpness", self.min_sharpness))
        thresholds_resolved = dict(thresholds)
        if intent == "factual":
            thresholds_resolved["min_clip_sim_factual"] = float(min_clip_gate)
        else:
            thresholds_resolved["min_clip_sim_conceptual"] = float(min_clip_gate)

        # Pre-filter (resolution, existence)
        candidate_meta: List[dict] = []
        kept_meta: List[dict] = []
        kept_paths: List[str] = []
        dims: List[Tuple[int, int]] = []
        for p in paths:
            meta = {"source": p}
            try:
                if not os.path.exists(p) or os.path.getsize(p) == 0:
                    meta["decision"] = "missing_or_empty"
                    candidate_meta.append(meta)
                    continue
                w, h = self._read_size(p)
                meta["resolution"] = (w, h)
                # Orientation-agnostic resolution gate: require short-side and long-side minimums
                short_side = min(w, h)
                long_side = max(w, h)
                req_short = min(min_w, min_h)
                req_long = max(min_w, min_h)
                short_ratio = float(short_side) / float(req_short or 1)
                long_ratio = float(long_side) / float(req_long or 1)
                resolution_ratio = min(short_ratio, long_ratio)
                meta["resolution_ratio"] = round(resolution_ratio, 4)
                meta["resolution_short_ratio"] = round(short_ratio, 4)
                meta["resolution_long_ratio"] = round(long_ratio, 4)
                meta["resolution_relaxed"] = short_ratio < 1.0 or long_ratio < 1.0
                if resolution_ratio < self.resolution_relax_ratio:
                    meta["decision"] = "reject:low_resolution"
                    candidate_meta.append(meta)
                    continue
                meta["path"] = p
                kept_paths.append(p)
                dims.append((w, h))
                kept_meta.append(meta)
                candidate_meta.append(meta)
            except Exception:
                meta["decision"] = "unreadable"
                candidate_meta.append(meta)

        if not kept_paths:
            return SelectionResult(
                path=None,
                score=0.0,
                note=(
                    "all rejected by resolution "
                    f"(min_short={min(min_w,min_h)} min_long={max(min_w,min_h)} relax_ratio={self.resolution_relax_ratio:.2f})"
                ),
                decision="rejected_all",
                metrics={"candidates": candidate_meta, "intent": intent},
            )

        # Semantic references
        if not (semantic_text and isinstance(semantic_text, str) and semantic_text.strip()):
            semantic_text = " ".join((visual_query or "").strip().split())
        if not (semantic_text_variants and isinstance(semantic_text_variants, list) and semantic_text_variants):
            semantic_text_variants = [f"{semantic_text} close-up shot", f"{semantic_text} wide scene"]
        if not (negative_semantic_texts and isinstance(negative_semantic_texts, list)):
            negative_semantic_texts = ["movie poster text", "vector illustration", "CGI render", "watermark"]

        positive_refs = [" ".join((semantic_text or "").split())]
        for v in (semantic_text_variants or [])[:4]:
            vv = " ".join((v or "").split())
            if vv:
                positive_refs.append(vv)

        def _default_candidate_metrics() -> Dict[str, object]:
            return {
                "ocr_chars": 0,
                "sharpness_norm": 0.0,
                "sharpness_var": 0.0,
                "faces": 0,
                "abstract_flag": False,
            }

        metrics_by_index: List[Dict[str, object]] = [_default_candidate_metrics() for _ in kept_paths]

        def _compute_candidate_metrics(index: int, asset_path: str) -> Tuple[int, Dict[str, object]]:
            try:
                ocr_chars_val = int(self._ocr_char_count(asset_path))
                sharp_norm_val, sharp_raw_val = self._sharpness_metrics(asset_path)
                faces_val = int(self._detect_faces(asset_path))
                abstract_flag_val = bool(self._is_vector_pattern(asset_path))
                return index, {
                    "ocr_chars": ocr_chars_val,
                    "sharpness_norm": float(sharp_norm_val),
                    "sharpness_var": float(sharp_raw_val),
                    "faces": faces_val,
                    "abstract_flag": abstract_flag_val,
                }
            except Exception:
                return index, _default_candidate_metrics()

        workers_env = os.getenv("ASSET_SELECTOR_WORKERS")
        try:
            configured_workers = int(workers_env) if workers_env else 0
        except ValueError:
            configured_workers = 0
        if configured_workers < 0:
            configured_workers = 0

        max_workers = configured_workers or min(4, len(kept_paths))
        max_workers = max(1, min(max_workers, len(kept_paths)))

        if max_workers == 1:
            for idx, path in enumerate(kept_paths):
                _, stats = _compute_candidate_metrics(idx, path)
                metrics_by_index[idx] = stats
        else:
            with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="asset_metrics") as executor:
                future_map = {
                    executor.submit(_compute_candidate_metrics, idx, path): idx
                    for idx, path in enumerate(kept_paths)
                }
                for future in as_completed(future_map):
                    idx = future_map[future]
                    try:
                        _, stats = future.result()
                    except Exception:
                        stats = _default_candidate_metrics()
                    metrics_by_index[idx] = stats
        effective_relax = max(float(relax_clip_margin or 0.0), float(relax_env or 0.0))
        scoring_context = AssetScoringContext(
            intent=intent,
            thresholds=thresholds_resolved,
            limits=limits,
            weights=weights_resolved,
            overlay_gate=float(overlay_gate),
            max_ocr_chars=max_ocr_chars,
            min_sharpness=float(min_sharpness_gate),
            relax_clip_margin=effective_relax,
        )

        candidates: List[AssetCandidate] = []
        for i, p in enumerate(kept_paths):
            w, h = dims[i]
            stats = metrics_by_index[i] or _default_candidate_metrics()
            res_ratio = float(kept_meta[i].get("resolution_ratio", min(1.5, (w * h) / float(1920 * 1080))))
            candidates.append(
                AssetCandidate(
                    path=p,
                    provider=kept_meta[i].get("provider"),
                    variant=kept_meta[i].get("variant"),
                    width=w,
                    height=h,
                    semantic_text=positive_refs[0] if positive_refs else semantic_text,
                    prompt=" | ".join(positive_refs) if positive_refs else semantic_text,
                    intent=intent,
                    ocr_chars=int(stats.get("ocr_chars", 0)),
                    sharpness_norm=float(stats.get("sharpness_norm", 0.0)),
                    sharpness_var=float(stats.get("sharpness_var", 0.0)),
                    faces=int(stats.get("faces", 0)),
                    abstract_flag=bool(stats.get("abstract_flag", False)),
                    resolution_ratio=res_ratio,
                )
            )

        results: List[AssetScoreResult] = []
        for cand in candidates:
            results.append(
                score_candidate(
                    candidate=cand,
                    positive_refs=positive_refs,
                    negative_refs=negative_semantic_texts or [],
                    models=self.scoring_models,
                    context=scoring_context,
                )
            )

        accepted: List[AssetScoreResult] = []
        relaxed: List[AssetScoreResult] = []

        for res, meta in zip(results, kept_meta):
            strict_ok, reason = passes_thresholds(res, scoring_context, allow_relax=False)
            meta.update(
                {
                    "clip_sim": round(res.clip_raw, 4),
                    "reward": round(res.reward_raw, 4),
                    "quality": round(res.quality_raw, 4),
                    "aesthetic": round(res.aesthetic_raw, 4),
                    "clip_norm": round(res.clip_norm, 4),
                    "reward_norm": round(res.reward_norm, 4),
                    "quality_norm": round(res.quality_norm, 4),
                    "aesthetic_norm": round(res.aesthetic_norm, 4),
                    "ocr_chars": int(res.candidate.ocr_chars),
                    "ocr_ratio": round(res.ocr_ratio, 4),
                    "sharpness_norm": round(res.sharpness_norm, 4),
                    "sharpness_var": round(res.candidate.sharpness_var, 2),
                    "faces": int(res.candidate.faces),
                    "abstract_flag": res.candidate.abstract_flag,
                    "resolution_ratio": round(res.candidate.resolution_ratio, 4),
                    "score": round(res.score, 4),
                    "decision": reason if not strict_ok else "accepted",
                    "neg_max": round(res.neg_max, 4),
                }
            )
            if strict_ok:
                res.decision = "accepted"
                accepted.append(res)
            else:
                relax_ok, relax_reason = passes_thresholds(res, scoring_context, allow_relax=relax_on_failure)
                if relax_ok:
                    res.decision = "relaxed_clip" if relax_reason.startswith("reject:clip") else relax_reason
                    relaxed.append(res)
                else:
                    res.decision = reason

        if not accepted and relaxed:
            relaxed.sort(key=lambda r: r.score, reverse=True)
            chosen = relaxed[0]
            note = f"relaxed clip={chosen.clip_raw:.2f} intent={intent}"
            metrics = {
                "candidates": candidate_meta,
                "intent": intent,
                "positive_refs": positive_refs,
                "negative_refs": negative_semantic_texts,
                "grounded_entities": grounded_entities or [],
            }
            return SelectionResult(
                path=chosen.candidate.path,
                score=float(chosen.score),
                note=note,
                decision=chosen.decision,
                metrics=metrics,
            )

        if not accepted:
            return SelectionResult(
                path=None,
                score=0.0,
                note=f"all rejected intent={intent} min_clip={min_clip_gate:.2f}",
                decision="rejected_all",
                metrics={"candidates": candidate_meta, "intent": intent},
            )

        accepted.sort(key=lambda r: r.score, reverse=True)
        best = accepted[0]
        note = f"accepted clip={best.clip_raw:.2f} score={best.score:.2f} intent={intent}"
        return SelectionResult(
            path=best.candidate.path,
            score=float(best.score),
            note=note,
            decision=best.decision,
            metrics={
                "chosen": kept_meta[results.index(best)],
                "candidates": candidate_meta,
                "intent": intent,
                "positive_refs": positive_refs,
                "negative_refs": negative_semantic_texts,
                "grounded_entities": grounded_entities or [],
                "target_duration_sec": target_duration_sec,
            },
        )
