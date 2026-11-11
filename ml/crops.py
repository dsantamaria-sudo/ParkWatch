"""Per-spot crop geometry, shared by build_dataset, training and inference.

One example = bounding box of the spot polygon (rois.json, points in %) enlarged by a
`margin` on each side, made square with letterboxing (no distortion) and resized.

The dataset stores each crop with a generous margin (STORE_MARGIN) plus the position of
the polygon bbox inside it; the actual margin and jitter are applied at load time. This
way `crop_margin` and `jitter` are hyperparameters that do not require rebuilding the
dataset, and cropping the stored crop gives exactly the same result as cropping the photo.
"""
from __future__ import annotations

import random

from PIL import Image, ImageDraw

STORE_MARGIN = 0.25  # >= largest crop_margin + jitter we expect to try
LETTERBOX_FILL = (114, 114, 114)

Box = tuple[float, float, float, float]  # x0, y0, x1, y1 in pixels


def polygon_bbox(points_pct: list[list[float]], width: int, height: int) -> Box:
    xs = [p[0] / 100 * width for p in points_pct]
    ys = [p[1] / 100 * height for p in points_pct]
    return min(xs), min(ys), max(xs), max(ys)


def expand_box(box: Box, margin: float, jitter: float = 0.0, rng: random.Random | None = None) -> Box:
    """Enlarges `box` by `margin` of its width/height on each side.

    With jitter > 0 (training only) the centre is shifted and the size scaled at random,
    by up to +-jitter of that width/height.
    """
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half_w, half_h = w * (0.5 + margin), h * (0.5 + margin)
    if jitter > 0:
        rng = rng or random
        cx += rng.uniform(-jitter, jitter) * w
        cy += rng.uniform(-jitter, jitter) * h
        scale = rng.uniform(1 - jitter, 1 + jitter)
        half_w, half_h = half_w * scale, half_h * scale
    return cx - half_w, cy - half_h, cx + half_w, cy + half_h


def clip_box(box: Box, width: int, height: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    x0, y0 = max(0, int(round(x0))), max(0, int(round(y0)))
    x1, y1 = min(width, int(round(x1))), min(height, int(round(y1)))
    return x0, y0, max(x0 + 1, x1), max(y0 + 1, y1)


def letterbox(image: Image.Image, size: int) -> Image.Image:
    """Scales the longer side to `size` and centres it on a grey square canvas (no distortion).
    A mask (mode "L") gets a black (0 = outside the spot) canvas."""
    scale = size / max(image.width, image.height)
    w, h = max(1, round(image.width * scale)), max(1, round(image.height * scale))
    canvas = Image.new(image.mode, (size, size), LETTERBOX_FILL if image.mode == "RGB" else 0)
    canvas.paste(image.resize((w, h), Image.BILINEAR), ((size - w) // 2, (size - h) // 2))
    return canvas


def crop_spot(image: Image.Image, points_pct: list[list[float]], margin: float) -> Image.Image:
    """Crop of one spot from the full photo (inference / build_dataset)."""
    box = expand_box(polygon_bbox(points_pct, image.width, image.height), margin)
    return image.crop(clip_box(box, image.width, image.height))


# --- Spot polygon: mask as a 4th input channel (B) / grey outside the spot (C) ------------
# The crops are stored in the reference geometry (every spot already aligned), so the polygon
# of rois.json, shifted to the stored crop, is exactly the spot inside it.

def spot_polygons(rois: dict) -> dict[str, list[list[float]]]:
    """{spot: polygon in reference-photo px} (rois.json points are in %)."""
    w, h = rois["image_size"]
    return {s["id"]: [[x / 100 * w, y / 100 * h] for x, y in s["points"]] for s in rois["spots"]}


def polygon_mask(size: tuple[int, int], points_px: list[list[float]], offset: tuple[float, float] = (0, 0)) -> Image.Image:
    """Mode "L" image of `size`: 255 inside the polygon (shifted by -offset), 0 outside."""
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).polygon([(x - offset[0], y - offset[1]) for x, y in points_px], fill=255)
    return mask


def grey_outside(image: Image.Image, mask: Image.Image) -> Image.Image:
    """Everything outside the spot painted with the letterbox grey (experiment C)."""
    return Image.composite(image, Image.new("RGB", image.size, LETTERBOX_FILL), mask)
