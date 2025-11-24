# src/media/aesthetic_scorer.py
from __future__ import annotations

from PIL import Image


class AestheticScorer:
    def __init__(self) -> None:
        self._clip = None
        self._fallback_text = None

    def _ensure_clip(self):
        if self._clip is not None:
            return
        from sentence_transformers import SentenceTransformer  # lazy

        self._clip = SentenceTransformer("clip-ViT-B-32")
        self._fallback_text = self._clip.encode(
            ["award-winning documentary photo, natural light, detailed, sharp"],
            convert_to_tensor=True,
            show_progress_bar=False,
        )[0]

    def score(self, path: str) -> float:
        try:
            from aesthetic_predictor import LAIONAestheticPredictor  # type: ignore

            predictor = LAIONAestheticPredictor()
            img = Image.open(path).convert("RGB")
            s = float(predictor.predict(img))
            return max(0.0, min(1.0, s / 10.0))
        except Exception:
            try:
                self._ensure_clip()
                img = Image.open(path).convert("RGB")
                im_emb = self._clip.encode([img], convert_to_tensor=True, show_progress_bar=False)[0]
                import torch

                sim = torch.cos_sim(im_emb, self._fallback_text).item()
                return float((sim + 1.0) / 2.0)
            except Exception:
                return 0.5
