from __future__ import annotations
import numpy as np
from PIL import Image

try:
    import pytesseract  # type: ignore
except Exception:
    pytesseract = None

class OcrTextGuard:
    def __init__(self, max_ratio: float = 0.12) -> None:
        self.max_ratio = max_ratio

    def reject_for_text_overlay(self, path: str) -> bool:
        if pytesseract is None:
            return False
        try:
            img = Image.open(path).convert("L")
            arr = np.array(img)
            thr = np.percentile(arr, 85)
            _ = (arr < thr).astype(np.uint8)
            data = pytesseract.image_to_data(img, output_type="dict")
            n_boxes = len(data.get("text", []))
            if n_boxes == 0:
                return False
            valid = sum(1 for t in data["text"] if isinstance(t, str) and t.strip() and any(c.isalpha() for c in t))
            ratio = valid / float(n_boxes)
            return ratio > self.max_ratio
        except Exception:
            return False
