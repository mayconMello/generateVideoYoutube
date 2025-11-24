from __future__ import annotations
from PIL import Image

try:
    import imagehash
except Exception:
    imagehash = None

class PerceptualHasher:
    def __init__(self, threshold: int = 6) -> None:
        self.threshold = threshold

    def similar(self, a_path: str, b_path: str) -> bool:
        if imagehash is None:
            return False
        try:
            with Image.open(a_path) as ia, Image.open(b_path) as ib:
                ha = imagehash.phash(ia.convert("RGB"))
                hb = imagehash.phash(ib.convert("RGB"))
            return ha - hb <= self.threshold
        except Exception:
            return False
