import glob
import math
import os
import random
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFilter


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _rand_color_pair() -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    palettes = [
        ((10, 32, 61), (26, 83, 92)),
        ((28, 27, 38), (79, 99, 149)),
        ((19, 39, 79), (163, 189, 237)),
        ((20, 20, 20), (60, 60, 60)),
        ((18, 45, 38), (27, 94, 89)),
        ((44, 22, 60), (102, 78, 131)),
        ((15, 25, 35), (35, 55, 75)),
    ]
    return random.choice(palettes)


def _make_gradient_noise_wallpaper(
        size: Tuple[int, int] = (1920, 1080),
        vignette: bool = True,
        blur_radius: float = 1.5,
        noise_strength: float = 0.06,
) -> Image.Image:
    w, h = size
    c0, c1 = _rand_color_pair()

    img = Image.new("RGB", (w, h))
    px = img.load()
    cx, cy = w / 2.0, h / 2.0
    max_d = math.hypot(cx, cy)

    ang = random.uniform(0, math.pi * 2)
    lx = math.cos(ang)
    ly = math.sin(ang)

    for y in range(h):
        for x in range(w):

            t_lin = (x * lx + y * ly) / (max(w, h))
            t_lin = (t_lin + 0.5)

            dx, dy = x - cx, y - cy
            d = math.hypot(dx, dy) / max_d
            t_rad = 1.0 - d
            t = _clamp(0.65 * t_lin + 0.35 * t_rad, 0.0, 1.0)

            r = int(_lerp(c0[0], c1[0], t))
            g = int(_lerp(c0[1], c1[1], t))
            b = int(_lerp(c0[2], c1[2], t))

            if noise_strength > 0:
                n = (random.random() - 0.5) * 2.0 * 255 * noise_strength
                r = int(_clamp(r + n, 0, 255))
                g = int(_clamp(g + n, 0, 255))
                b = int(_clamp(b + n, 0, 255))
            px[x, y] = (r, g, b)

    if vignette:

        vign = Image.new("L", (w, h))
        dv = vign.load()
        for y in range(h):
            for x in range(w):
                dx, dy = x - cx, y - cy
                d = math.hypot(dx, dy) / max_d
                shade = int(_clamp(255 * (1.0 - d ** 1.6), 0, 255))
                dv[x, y] = shade
        vign = vign.filter(ImageFilter.GaussianBlur(radius=0.5 * min(w, h) / 50))
        img = Image.composite(img, Image.new("RGB", (w, h), (0, 0, 0)), vign.point(lambda p: 255 - p))

    if blur_radius > 0:
        img = img.filter(ImageFilter.GaussianBlur(blur_radius))
    return img


def _annotate_subtle_guides(img: Image.Image) -> Image.Image:
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size
    alpha = 10
    color = (255, 255, 255, alpha)
    step = random.choice([h // 9, h // 11, h // 13])

    for y in range(step, h, step):
        draw.line([(0, y), (w, y)], fill=color, width=1)
    return img


def ensure_fallbacks(count: int = 6, size: Tuple[int, int] = (1920, 1080)) -> List[str]:
    env_dir = os.getenv("FALLBACKS_DIR")

    def _collect_images(folder: str) -> List[str]:
        exts = ("*.jpg", "*.jpeg", "*.png", "*.webp")
        files: List[str] = []
        for pat in exts:
            files.extend(glob.glob(os.path.join(folder, pat)))
        return sorted(files)

    if env_dir and os.path.isdir(env_dir):
        imgs = _collect_images(env_dir)
        if imgs:
            return imgs

    local_dir = _ensure_dir(str(Path("./assets/fallbacks").resolve()))
    existing = _collect_images(local_dir)
    if len(existing) >= count:
        return existing

    to_make = max(0, count - len(existing))
    made: List[str] = []
    for i in range(to_make):
        img = _make_gradient_noise_wallpaper(size=size)
        if random.random() < 0.35:
            img = _annotate_subtle_guides(img)
        out = os.path.join(local_dir, f"fallback_{len(existing) + i:02d}.jpg")
        img.save(out, quality=92, subsampling=1, optimize=True)
        made.append(out)

    return _collect_images(local_dir)
