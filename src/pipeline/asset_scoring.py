from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _normalize_reward(value: float) -> float:
    return _clamp01((value + 10.0) / 20.0)


def _normalize_clip(value: float) -> float:
    return _clamp01(value)


def _normalize_quality(value: float) -> float:
    return _clamp01(value / 10.0)


def _normalize_aesthetic(value: float) -> float:
    return _clamp01(value / 5.0)


@dataclass
class AssetCandidate:
    path: str
    provider: Optional[str]
    variant: Optional[str]
    width: int
    height: int
    semantic_text: str
    prompt: str
    intent: str
    ocr_chars: int
    sharpness_norm: float
    sharpness_var: float
    faces: int
    abstract_flag: bool
    resolution_ratio: float


@dataclass
class AssetScoringContext:
    intent: str
    thresholds: Dict[str, float]
    limits: Dict[str, float]
    weights: Dict[str, float]
    overlay_gate: float
    max_ocr_chars: int
    min_sharpness: float
    relax_clip_margin: float


@dataclass
class AssetScoreResult:
    candidate: AssetCandidate
    score: float
    clip_raw: float
    reward_raw: float
    quality_raw: float
    aesthetic_raw: float
    clip_norm: float
    reward_norm: float
    quality_norm: float
    aesthetic_norm: float
    ocr_ratio: float
    sharpness_norm: float
    decision: str
    neg_max: float


class AssetScoringModels:
    def __init__(self) -> None:
        self._aesthetic = None
        self._reward = None
        self._quality = None

    def clip_score(self, image_path: str, text: str, fallback_query: str) -> float:
        try:
            from src.validation.semantic_validator import score_image_path

            return float(score_image_path(image_path, text, fallback_query))
        except Exception:
            return 0.0

    def clip_score_with_refs(self, image_path: str, positive_refs: List[str]) -> float:
        try:
            from src.validation.semantic_validator import score_images_with_refs
        except Exception:
            return self.clip_score(image_path, " ".join(positive_refs), " ".join(positive_refs))

        try:
            scores = score_images_with_refs([image_path], positive_refs, [], agg="max")
            if scores:
                return float(scores[0])
            return 0.0
        except Exception:
            return 0.0

    def clip_negative_score(self, image_path: str, negative_refs: List[str]) -> float:
        if not negative_refs:
            return 0.0
        try:
            from src.validation.semantic_validator import score_images_with_refs
        except Exception:
            return 0.0
        try:
            scores = score_images_with_refs([image_path], negative_refs, [], agg="max")
            if scores:
                return float(scores[0])
            return 0.0
        except Exception:
            return 0.0

    def reward_score(self, image_path: str, prompt: str) -> float:
        if self._reward is None:
            try:
                from imagereward import ImageReward

                self._reward = ImageReward("ImageReward/ImageReward-v1.0")
            except Exception:
                self._reward = False
        if self._reward is False:
            return 0.0
        try:
            return float(self._reward.score(prompt, image_path))
        except Exception:
            return 0.0

    def aesthetic_score(self, image_path: str) -> float:
        model = self._get_aesthetic_model()
        if model is None:
            return 0.0
        try:
            result = model(image_path)
            if not result:
                return 0.0
            first = result[0] if isinstance(result, list) else result
            return float(first.get("score", 0.0) if isinstance(first, dict) else 0.0)
        except Exception:
            return 0.0

    def quality_score(self, image_path: str) -> float:
        model = self._get_quality_model()
        if model is None:
            return 0.0
        try:
            result = model(image_path)
            if not result:
                return 0.0
            first = result[0] if isinstance(result, list) else result
            return float(first.get("score", 0.0) if isinstance(first, dict) else 0.0)
        except Exception:
            return 0.0

    def _get_aesthetic_model(self):
        if self._aesthetic is False:
            return None
        if self._aesthetic is None:
            try:
                from transformers import pipeline

                self._aesthetic = pipeline("image-classification", model="rsinema/aesthetic-scorer")
            except Exception:
                self._aesthetic = False
        return self._aesthetic if self._aesthetic is not False else None

    def _get_quality_model(self):
        if self._quality is False:
            return None
        if self._quality is None:
            try:
                from transformers import pipeline

                self._quality = pipeline("image-quality-assessment", model="matthewyuan/image-quality-fusion")
            except Exception:
                self._quality = False
        return self._quality if self._quality is not False else None


def score_candidate(
    *,
    candidate: AssetCandidate,
    positive_refs: List[str],
    negative_refs: List[str],
    models: AssetScoringModels,
    context: AssetScoringContext,
) -> AssetScoreResult:
    prompt = candidate.prompt or candidate.semantic_text
    clip_raw = models.clip_score_with_refs(candidate.path, positive_refs) if positive_refs else models.clip_score(
        candidate.path, prompt, prompt
    )
    neg_max = models.clip_negative_score(candidate.path, negative_refs) if negative_refs else 0.0
    reward_raw = models.reward_score(candidate.path, prompt)
    quality_raw = models.quality_score(candidate.path)
    aesthetic_raw = models.aesthetic_score(candidate.path)

    clip_norm = _normalize_clip(clip_raw)
    reward_norm = _normalize_reward(reward_raw)
    quality_norm = _normalize_quality(quality_raw)
    aesthetic_norm = _normalize_aesthetic(aesthetic_raw)

    if context.intent == "conceptual":
        base = (
            0.30 * clip_norm
            + 0.30 * reward_norm
            + 0.25 * aesthetic_norm
            + 0.15 * quality_norm
        )
    else:
        base = (
            0.45 * clip_norm
            + 0.25 * reward_norm
            + 0.20 * quality_norm
            + 0.10 * aesthetic_norm
        )

    ocr_ratio = float(candidate.ocr_chars) / float(max(context.max_ocr_chars, 1)) if context.max_ocr_chars else 0.0

    neg_penalty_k = float(os.getenv("CLIP_NEG_PENALTY", "0.08"))
    if context.intent == "conceptual":
        neg_penalty_k *= 0.7

    score = base
    score -= context.weights.get("ocr_penalty", 0.0) * ocr_ratio
    score += context.weights.get("face_bonus", 0.0) * (1.0 if candidate.faces > 0 else 0.0)
    score -= context.weights.get("abstract_penalty", 0.0) * (1.0 if candidate.abstract_flag else 0.0)
    score += context.weights.get("resolution_score", 0.0) * candidate.resolution_ratio
    score += context.weights.get("sharpness", 0.0) * candidate.sharpness_norm
    score -= neg_penalty_k * neg_max

    return AssetScoreResult(
        candidate=candidate,
        score=float(score),
        clip_raw=float(clip_raw),
        reward_raw=float(reward_raw),
        quality_raw=float(quality_raw),
        aesthetic_raw=float(aesthetic_raw),
        clip_norm=clip_norm,
        reward_norm=reward_norm,
        quality_norm=quality_norm,
        aesthetic_norm=aesthetic_norm,
        ocr_ratio=ocr_ratio,
        sharpness_norm=float(candidate.sharpness_norm),
        decision="pending",
        neg_max=float(neg_max),
    )


def passes_thresholds(result: AssetScoreResult, context: AssetScoringContext, *, allow_relax: bool) -> Tuple[bool, str]:
    clip_gate = (
        float(context.thresholds.get("min_clip_sim_conceptual", 0.0))
        if context.intent == "conceptual"
        else float(context.thresholds.get("min_clip_sim_factual", 0.0))
    )
    clip_ok = result.clip_raw >= clip_gate
    if not clip_ok and allow_relax:
        clip_ok = result.clip_raw >= max(0.0, clip_gate - context.relax_clip_margin)
    if not clip_ok:
        return False, "reject:clip"

    if result.ocr_ratio > context.overlay_gate:
        return False, "reject:overlay"
    if result.sharpness_norm < context.min_sharpness:
        return False, "reject:soft"

    quality_min = float(context.limits.get("quality_min", 0.0) or 0.0)
    if quality_min and result.quality_norm < quality_min:
        return False, "reject:quality"

    aesthetic_min = float(context.limits.get("aesthetic_min", 0.0) or 0.0)
    if aesthetic_min and result.aesthetic_norm < aesthetic_min:
        return False, "reject:aesthetic"

    return True, "accepted"
