import hashlib
import os
import uuid
from dataclasses import dataclass
from typing import List, Optional, Tuple

try:
    import cv2
except Exception:
    cv2 = None
import numpy as np
try:
    import pytesseract
except Exception:
    pytesseract = None
import requests
from PIL import Image, ImageOps

from src.validation.semantic_validator import (
    score_images_batch,
    score_image_path,
    score_images_with_refs,
)


@dataclass
class PickResult:
    path: Optional[str]
    score: float
    note: str = ""
    metrics: Optional[dict] = None
    decision: Optional[str] = None


def _md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")


def _download_to(local_path: str, url: str) -> str:
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(65536):
                if chunk:
                    f.write(chunk)
    return local_path


class SmartAssetPicker:
    """
    Ranks image candidates using semantic similarity (semantic_validator) + visual heuristics.
    No direct SentenceTransformer usage here; the semantic_validator module handles that.

    Heuristics considered:
      - Resolution gates/bonus
      - Sharpness (Laplacian variance)
      - OCR text (penalty as "overlay")
      - Face bonus
      - Abstract/vector-pattern penalty
    """

    def __init__(
            self,
            cache_dir: str,
            min_width: int = 960,
            min_height: int = 540,
            default_clip_threshold: float = 0.27,
            default_overlay_limit: float = 0.10,
            edge_density_max: float = 0.16,
            ocr_text_chars_max: int = 8,
    ):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        self.min_width = int(min_width)
        self.min_height = int(min_height)
        self.default_clip_threshold = float(default_clip_threshold)
        self.default_overlay_limit = float(default_overlay_limit)
        self.edge_density_max = float(edge_density_max)
        self.ocr_text_chars_max = int(ocr_text_chars_max)

    @staticmethod
    def _read_size(path: str) -> Tuple[int, int]:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            return im.size

    def _ensure_local(self, candidate: str) -> str:
        if _is_url(candidate):
            loc = os.path.join(self.cache_dir, f"{uuid.uuid4().hex}.jpg")
            _download_to(loc, candidate)
            return loc
        return candidate

    def _ocr_char_count(self, path: str) -> int:
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

    def _load_face_detector(self):
        if cv2 is None:
            return None
        if getattr(self, "_face_detector", None) is False:
            return None
        detector = getattr(self, "_face_detector", None)
        if detector is None:
            try:
                cascade_dir = getattr(cv2, "data", None)
                if cascade_dir is None:
                    self._face_detector = False
                    return None
                cascade_path = os.path.join(cascade_dir.haarcascades, "haarcascade_frontalface_default.xml")
                if not os.path.exists(cascade_path):
                    self._face_detector = False
                    return None
                detector = cv2.CascadeClassifier(cascade_path)
                if detector.empty():
                    self._face_detector = False
                    return None
                self._face_detector = detector
            except Exception:
                self._face_detector = False
                return None
        return detector

    def _detect_faces(self, path: str) -> int:
        detector = self._load_face_detector()
        if detector is None:
            return 0
        try:
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

    def pick(
            self,
            scene_text: str,
            visual_query: str,
            paths: List[str],
            scene_intent: Optional[str] = None,
            clip_threshold: Optional[float] = None,
            overlay_limit: Optional[float] = None,
            **kwargs,
    ) -> PickResult:
        """
        Ranks images with semantic_validator (CLIP) + heuristics and returns the best path.
        Signature preserved: scene_text, visual_query, paths, scene_intent, clip_threshold, overlay_limit, **kwargs
        """
        if not paths:
            return PickResult(
                path=None,
                score=0.0,
                note="no candidates",
                metrics={"candidates": []},
                decision="empty",
            )

        relax_on_failure = bool(kwargs.get("relax_on_failure", False))
        try:
            relax_clip_margin = float(kwargs.get("relax_clip_margin", 0.05))
        except (TypeError, ValueError):
            relax_clip_margin = 0.05

        intent = (scene_intent or "factual").strip().lower()
        if intent not in {"factual", "conceptual"}:
            intent = "factual"

        booster_map = {
            "factual": "documentary realism, real-world footage, no watermark, no captions",
            "conceptual": "cinematic concept art, atmospheric mood, no watermark, no captions",
        }
        boosted_scene_text = f"{scene_text.strip()} | {booster_map[intent]}"

        policy_weights = kwargs.get("weights") or {}
        weights_cfg = policy_weights.get(intent) or {}

        def _w(name: str, default: float) -> float:
            try:
                return float(weights_cfg.get(name, default))
            except (TypeError, ValueError):
                return float(default)

        weights = {
            "clip_sim": _w("clip_sim", 1.0),
            "resolution_score": _w("resolution_score", 0.30),
            "sharpness": _w("sharpness", 0.20),
            "ocr_penalty": _w("ocr_penalty", 0.80),
            "face_bonus": _w("face_bonus", 0.15),
            "abstract_penalty": _w("abstract_penalty", 0.40),
        }

        thresholds_cfg = kwargs.get("thresholds") or {}
        min_res_cfg = thresholds_cfg.get("min_resolution") or [self.min_width, self.min_height]
        try:
            min_w = int(min_res_cfg[0])
            min_h = int(min_res_cfg[1])
        except Exception:
            min_w, min_h = self.min_width, self.min_height
        min_w = max(min_w, 1)
        min_h = max(min_h, 1)

        # Resolve CLIP gates via policy + ENV (when present)
        if intent == "factual":
            policy_gate = float(thresholds_cfg.get("min_clip_sim_factual", self.default_clip_threshold))
            env_gate = float(os.getenv("CLIP_FACTUAL_GATE_IMG", str(policy_gate)))
            min_clip_default = max(policy_gate, env_gate)
        else:
            policy_gate = float(
                thresholds_cfg.get("min_clip_sim_conceptual", max(0.18, self.default_clip_threshold - 0.05))
            )
            env_gate = float(os.getenv("CLIP_CONCEPTUAL_GATE_IMG", str(policy_gate)))
            min_clip_default = max(policy_gate, env_gate)
        min_clip_gate = float(clip_threshold) if clip_threshold is not None else min_clip_default
        relax_margin_env = float(os.getenv("CLIP_RELAX_MARGIN", "0.02"))

        if overlay_limit is not None:
            overlay_gate = float(overlay_limit)
        elif "overlay_limit" in thresholds_cfg:
            overlay_gate = float(thresholds_cfg.get("overlay_limit"))
        else:
            overlay_gate = self.default_overlay_limit

        limits_cfg = kwargs.get("limits") or {}
        max_ocr_chars = int(limits_cfg.get("max_ocr_chars", self.ocr_text_chars_max))
        min_sharpness_gate = float(limits_cfg.get("min_sharpness", 0.18))

        local_paths: List[str] = []
        dims: List[Tuple[int, int]] = []
        candidate_meta: List[dict] = []

        for original in paths:
            meta = {"source": original}
            try:
                local = self._ensure_local(original)
            except Exception:
                meta["decision"] = "download_failed"
                candidate_meta.append(meta)
                continue

            meta["path"] = local
            if not os.path.exists(local) or os.path.getsize(local) == 0:
                meta["decision"] = "missing_or_empty"
                candidate_meta.append(meta)
                continue

            try:
                w, h = self._read_size(local)
            except Exception:
                meta["decision"] = "unreadable"
                candidate_meta.append(meta)
                continue

            meta["resolution"] = (w, h)
            # Orientation-agnostic resolution gate: require short-side and long-side minimums
            short_side = min(w, h)
            long_side = max(w, h)
            req_short = min(min_w, min_h)
            req_long = max(min_w, min_h)
            if short_side < req_short or long_side < req_long:
                meta["decision"] = "reject:low_resolution"
                candidate_meta.append(meta)
                continue

            local_paths.append(local)
            dims.append((w, h))
            candidate_meta.append(meta)

        if not local_paths:
            return PickResult(
                path=None,
                score=0.0,
                note=f"all rejected by resolution (min_short={min(min_w,min_h)} min_long={max(min_w,min_h)})",
                metrics={"candidates": candidate_meta, "intent": intent},
                decision="rejected_all",
            )

        # Semantic inputs from kwargs
        semantic_text: Optional[str] = kwargs.get("semantic_text")
        semantic_text_variants: Optional[List[str]] = kwargs.get("semantic_text_variants")
        negative_semantic_texts: Optional[List[str]] = kwargs.get("negative_semantic_texts")
        grounded_entities: Optional[List[str]] = kwargs.get("grounded_entities")

        # Fallbacks when LLM is missing fields
        if not (semantic_text and isinstance(semantic_text, str) and semantic_text.strip()):
            semantic_text = " ".join(visual_query.strip().split())
        if not (semantic_text_variants and isinstance(semantic_text_variants, list)):
            semantic_text_variants = [
                f"{semantic_text} close-up shot",
                f"{semantic_text} wide scene",
            ]
        if not (negative_semantic_texts and isinstance(negative_semantic_texts, list)):
            negative_semantic_texts = [
                "movie poster text",
                "vector illustration",
                "CGI render",
                "watermark",
            ]

        positive_refs = [" ".join((semantic_text or "").split())]
        for v in (semantic_text_variants or [])[:4]:
            vv = " ".join((v or "").split())
            if vv:
                positive_refs.append(vv)

        # Pre-ranking by CLIP using semantic references
        try:
            sims = score_images_with_refs(local_paths, positive_refs, negative_semantic_texts or [], agg="max")
        except Exception:
            sims = []
            for p in local_paths:
                try:
                    sims.append(float(score_image_path(p, boosted_scene_text, visual_query)))
                except Exception:
                    sims.append(0.0)
        # Negative penalty proxy: max similarity to any negative reference
        try:
            neg_max_scores = score_images_with_refs(local_paths, negative_semantic_texts or [], [], agg="max")
        except Exception:
            neg_max_scores = [0.0] * len(local_paths)
        neg_penalty_k = float(os.getenv("CLIP_NEG_PENALTY", "0.08"))
        if intent == "conceptual":
            neg_penalty_k *= 0.7

        accepted: List[Tuple[float, str, float, dict]] = []
        best_clip_candidate: Optional[dict] = None

        for i, p in enumerate(local_paths):
            w, h = dims[i]
            clip_sim = float(sims[i] if i < len(sims) else 0.0)

            ocr_chars = self._ocr_char_count(p)
            sharp_norm, sharp_raw = self._sharpness_metrics(p)
            faces = self._detect_faces(p)
            abstract_flag = bool(self._is_vector_pattern(p))

            ocr_ratio = float(ocr_chars) / float(max(max_ocr_chars, 1)) if max_ocr_chars else 0.0

            res_ratio = min(1.5, (w * h) / float(1920 * 1080))

            if clip_sim < min_clip_gate:
                decision = f"reject:clip<{min_clip_gate:.2f}"
            elif ocr_ratio > overlay_gate:
                decision = "reject:overlay"
            elif sharp_norm < min_sharpness_gate:
                decision = "reject:soft"
            else:
                decision = "accepted"

            neg_max = float(neg_max_scores[i] if i < len(neg_max_scores) else 0.0)
            score = (
                weights["clip_sim"] * clip_sim
                + weights["resolution_score"] * res_ratio
                + weights["sharpness"] * sharp_norm
                + weights["face_bonus"] * (1.0 if faces > 0 else 0.0)
                - weights["ocr_penalty"] * ocr_ratio
                - weights["abstract_penalty"] * (1.0 if abstract_flag else 0.0)
                - (neg_penalty_k * neg_max)
            )

            meta = candidate_meta[i]
            meta.update(
                {
                    "clip_sim": round(clip_sim, 4),
                    "ocr_chars": int(ocr_chars),
                    "ocr_ratio": round(ocr_ratio, 4),
                    "sharpness_norm": round(sharp_norm, 4),
                    "sharpness_var": round(sharp_raw, 2),
                    "faces": int(faces),
                    "abstract_flag": abstract_flag,
                    "resolution_ratio": round(res_ratio, 4),
                    "score": round(score, 4),
                    "decision": decision,
                    "neg_max": round(neg_max, 4),
                    "neg_penalty_k": neg_penalty_k,
                }
            )

            if decision == "accepted":
                accepted.append((score, p, clip_sim, meta))
            elif decision.startswith("reject:clip"):
                if best_clip_candidate is None or meta.get("clip_sim", 0.0) > best_clip_candidate.get("clip_sim", 0.0):
                    best_clip_candidate = meta

        if not accepted:

            if relax_on_failure and (relax_clip_margin > 0.0 or relax_margin_env > 0.0) and best_clip_candidate:
                clip_val = float(best_clip_candidate.get("clip_sim", 0.0))
                effective_relax = max(relax_clip_margin, relax_margin_env)
                res_ratio = float(best_clip_candidate.get("resolution_ratio", 0.0))
                if clip_val >= max(0.0, min_clip_gate - effective_relax) and res_ratio >= 0.8:
                    note = f"relaxed clip={clip_val:.2f} (< {min_clip_gate:.2f}) intent={intent}"
                    relaxed_metrics = {
                        "candidates": candidate_meta,
                        "relaxed": {
                            "reason": "clip",
                            "clip_sim": clip_val,
                            "threshold": min_clip_gate,
                            "margin": effective_relax,
                        },
                        "intent": intent,
                        "positive_refs": positive_refs,
                        "negative_refs": negative_semantic_texts,
                    }
                    return PickResult(
                        path=best_clip_candidate.get("path"),
                        score=float(best_clip_candidate.get("score", clip_val)),
                        note=note,
                        metrics=relaxed_metrics,
                        decision="relaxed_clip",
                    )

            note = f"all rejected intent={intent} min_clip={min_clip_gate:.2f}"
            return PickResult(
                path=None,
                score=0.0,
                note=note,
                metrics={"candidates": candidate_meta, "intent": intent},
                decision="rejected_all",
            )

        accepted.sort(key=lambda t: t[0], reverse=True)
        best_score, best_path, best_clip, best_meta = accepted[0]
        note = f"accepted clip={best_clip:.2f} score={best_score:.2f} intent={intent}"

        return PickResult(
            path=best_path,
            score=float(best_score),
            note=note,
            metrics={
                "chosen": best_meta,
                "candidates": candidate_meta,
                "intent": intent,
                "positive_refs": positive_refs,
                "negative_refs": negative_semantic_texts,
                "grounded_entities": grounded_entities or [],
            },
            decision="accepted",
        )
