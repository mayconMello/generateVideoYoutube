from __future__ import annotations

import os
from typing import Any, List, Tuple, cast

import cv2
import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer, util

PILImage = Image.Image

_CLIP_MODEL_NAME = "sentence-transformers/clip-ViT-B-32"

_clip_model: SentenceTransformer | None = None
_clip_device: str | None = None


def _resolve_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    try:
        idx = torch.cuda.current_device()
        major, _ = torch.cuda.get_device_capability(idx)
        return f"cuda:{idx}" if major >= 7 else "cpu"
    except ValueError:
        return "cpu"


def _clip() -> SentenceTransformer:
    global _clip_model, _clip_device
    if _clip_model is None:
        _clip_device = _resolve_device()

        try:
            _clip_model = SentenceTransformer(_CLIP_MODEL_NAME, device=_clip_device)
        except ValueError:
            _clip_device = "cpu"
            _clip_model = SentenceTransformer(_CLIP_MODEL_NAME, device=_clip_device)
    return _clip_model


def _sample_frames(
    video_path: str,
    *,
    every_n_frames: int = 12,
    max_frames: int = 32,
) -> List[PILImage]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    frames: List[PILImage] = []
    idx = 0
    try:
        while len(frames) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % max(1, every_n_frames) == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(rgb).copy())
            idx += 1
    finally:
        cap.release()
    return frames


def _encode_text_dual(
    model: SentenceTransformer,
    scene_text: str,
    visual_query: str,
    *,
    en_weight: float = 0.60,
) -> torch.Tensor:
    a = (scene_text or "").strip()
    b = (visual_query or "").strip()

    if a and b:
        embs = model.encode([a, b], convert_to_tensor=True, show_progress_bar=False)

        alpha = float(min(1.0, max(0.0, en_weight)))
        w1 = 1.0 - alpha
        w2 = alpha

        mixed = w1 * embs[0] + w2 * embs[1]
        return mixed
    elif a:
        return model.encode([a], convert_to_tensor=True, show_progress_bar=False)[0]
    elif b:
        return model.encode([b], convert_to_tensor=True, show_progress_bar=False)[0]
    else:
        return model.encode(["visual content"], convert_to_tensor=True, show_progress_bar=False)[0]


def validate_video_frames(
    video_path: str,
    scene_text: str,
    visual_query: str,
    *,
    clip_threshold: float = 0.30,
    every_n_frames: int = 12,
    max_frames: int = 32,
) -> Tuple[bool, float, float, str]:
    frames = _sample_frames(video_path, every_n_frames=every_n_frames, max_frames=max_frames)
    if not frames:
        return False, 0.0, 0.0, "no_frames"

    model = _clip()
    txt_emb = _encode_text_dual(model, scene_text, visual_query, en_weight=0.60)

    frames_typed: List[PILImage] = frames
    img_embs = model.encode(
        cast(List[Any], frames_typed),
        convert_to_tensor=True,
        show_progress_bar=False,
    )
    sims = util.cos_sim(txt_emb, img_embs)[0].detach().cpu().numpy().astype(np.float32)

    avg = float(np.mean(sims))
    mn = float(np.min(sims))
    ok = (avg >= clip_threshold * 0.95) and (mn >= clip_threshold * 0.70)
    reason = "ok" if ok else "low_similarity"
    return ok, avg, mn, reason


def score_image_path(image_path: str, scene_text: str, visual_query: str) -> float:
    if not os.path.exists(image_path):
        return 0.0
    try:
        img: PILImage = Image.open(image_path).convert("RGB").copy()
    except ValueError:
        return 0.0

    model = _clip()
    txt_emb = _encode_text_dual(model, scene_text, visual_query, en_weight=0.60)
    img_emb = model.encode([cast[Any, PILImage](img)], convert_to_tensor=True, show_progress_bar=False)[0]
    return float(util.cos_sim(txt_emb, img_emb).item())


def score_images_batch(image_paths: List[str], scene_text: str, visual_query: str) -> List[float]:
    paths = [p for p in image_paths if os.path.exists(p)]
    if not paths:
        return []

    model = _clip()
    txt_emb = _encode_text_dual(model, scene_text, visual_query, en_weight=0.60)

    valid_imgs: List[PILImage] = []
    valid_idx: List[int] = []
    for i, p in enumerate(paths):
        try:
            im: PILImage = Image.open(p).convert("RGB").copy()
            valid_imgs.append(im)
            valid_idx.append(i)
        except ValueError:
            pass

    if not valid_imgs:
        out = [0.0] * len(image_paths)
        return out

    img_embs = model.encode(
        cast(List[Any], valid_imgs),
        convert_to_tensor=True,
        show_progress_bar=False,
    )
    sims = util.cos_sim(txt_emb, img_embs)[0].detach().cpu().numpy().astype(np.float32)

    out = [0.0] * len(image_paths)
    for j, i in enumerate(valid_idx):
        out[i] = float(sims[j])
    return out


__all__ = [
    "_clip",
    "_resolve_device",
    "validate_video_frames",
    "score_image_path",
    "score_images_batch",
]


def score_images_with_refs(
    image_paths: List[str],
    positive_refs: List[str],
    negative_refs: List[str],
    *,
    agg: str = "max",
) -> List[float]:
    """Return positive CLIP scores for images given textual references.

    - Computes embeddings for `positive_refs` and each valid image.
    - For each image, returns the aggregated positive similarity:
      max(sim(img, ref_i)) when agg="max" (default) or the mean of top-2 if agg="top2-mean".
    - Negative refs are accepted for symmetry with selector contract but are NOT applied here.
      The selector is responsible for computing penalties using negative refs.

    The function is robust to unreadable paths and returns 0.0 for missing images.
    """
    paths = [p for p in image_paths if os.path.exists(p)]
    if not paths or not positive_refs:
        return [0.0] * len(image_paths)

    model = _clip()

    # Encode positives
    pos_texts = [" ".join((t or "").strip().split()) for t in positive_refs if (t or "").strip()]
    if not pos_texts:
        return [0.0] * len(image_paths)
    pos_embs = model.encode(pos_texts, convert_to_tensor=True, show_progress_bar=False)

    # Load valid images
    valid_imgs: List[PILImage] = []
    valid_idx: List[int] = []
    for i, p in enumerate(image_paths):
        if not os.path.exists(p):
            continue
        try:
            im: PILImage = Image.open(p).convert("RGB").copy()
            valid_imgs.append(im)
            valid_idx.append(i)
        except Exception:
            pass

    if not valid_imgs:
        return [0.0] * len(image_paths)

    img_embs = model.encode(valid_imgs, convert_to_tensor=True, show_progress_bar=False)

    out = [0.0] * len(image_paths)
    # Compute sim matrix [N_valid, N_pos]
    sims = util.cos_sim(img_embs, pos_embs).detach().cpu().numpy().astype(np.float32)
    for row, i in enumerate(valid_idx):
        vals = sims[row]
        if agg == "top2-mean" and len(vals) >= 2:
            top2 = np.partition(vals, -2)[-2:]
            score = float(np.mean(top2))
        else:
            score = float(np.max(vals))
        out[i] = score
    return out


# Keep exports updated
__all__.append("score_images_with_refs")
