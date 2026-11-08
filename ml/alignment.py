"""ParkWatch — alignment of every photo with the reference photo of the ROIs (camera drift).

The camera is glued in place and moves over time. align_photos.py tracks fixed landmarks of
the scene photo by photo (landmarks.py) and stores, for every photo, where each landmark is
and the warp of every SPOT (reference -> photo): the fisheye lens moves each part of the image
differently, so each spot is corrected with the landmarks around it. This module reads that
table (alignment.csv):

  - warps(blob): {spot: warp} of the photo (a photo missing from the table takes the nearest
    one in time; with an empty table, {} = no correction).
  - align_image(): straightens the photo to the geometry of the reference with one spot's
    warp, so that spot's polygon of rois.json applies as is. Same step in build_dataset and in
    inference.
  - transform_points(): a polygon of the reference moved onto the original photo (Label Studio).

No OpenCV: only numpy + PIL.
"""
from __future__ import annotations

import bisect
import csv
import math
import re
from datetime import datetime
from pathlib import Path

from PIL import Image

TABLE_FORMAT = 2   # 1: one warp per photo (SIFT on static regions); 2: landmarks + one warp per spot
BASE_COLUMNS = ["blob", "ts_utc", "status", "n_found", "found"]  # status: tracked | lost | anchor
WARP_KEYS = ["m00", "m01", "m02", "m10", "m11", "m12"]
IDENTITY = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
NAME_RE = re.compile(r"parkwatch_(\d{8}T\d{6}Z)\.jpg$")
SPOT_COL_RE = re.compile(r"^(p\d+)_m00$")

Warp = tuple[float, float, float, float, float, float]  # m00 m01 m02 / m10 m11 m12 (reference -> photo)


def blob_time(blob: str) -> datetime:
    return datetime.strptime(NAME_RE.search(blob).group(1), "%Y%m%dT%H%M%SZ")


def describe(warp: Warp) -> dict:
    """Shift (px), rotation (degrees) and scale of a warp."""
    m00, _, m02, m10, _, m12 = warp
    return {"dx": m02, "dy": m12, "angle": math.degrees(math.atan2(m10, m00)), "scale": math.hypot(m00, m10)}


def landmark_columns(landmarks) -> list[str]:
    keys = sorted(landmarks, key=lambda k: int(re.sub(r"\D", "", k) or 0))
    return [c for k in keys for c in (f"{k}_x", f"{k}_y")]


def spot_columns(spots) -> list[str]:
    return [f"{s}_{m}" for s in spots for m in WARP_KEYS]


class Alignment:
    def __init__(self, rows: list[dict]):
        self.rows = {r["blob"]: r for r in rows}
        self.spots = sorted({m.group(1) for r in rows[:1] for c in r if (m := SPOT_COL_RE.match(c))})
        self._times = sorted((blob_time(b), b) for b in self.rows)

    @classmethod
    def from_dir(cls, path: Path | None) -> "Alignment":
        """Table from a folder (align_photos output); without folder or table, empty (no correction)."""
        csv_path = Path(path) / "alignment.csv" if path else None
        if csv_path is None or not csv_path.exists():
            return cls([])
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if rows and "status" not in rows[0]:
            raise SystemExit(f"{csv_path}: table format 1 (one warp per photo); run align_photos again")
        return cls(rows)

    def _row(self, blob: str) -> dict | None:
        if blob in self.rows:
            return self.rows[blob]
        if not self._times:
            return None
        t = blob_time(blob)
        i = bisect.bisect_left(self._times, (t, ""))
        nearest = min((self._times[j] for j in (i - 1, i) if 0 <= j < len(self._times)), key=lambda x: abs(x[0] - t))
        return self.rows[nearest[1]]

    def warps(self, blob: str) -> dict[str, Warp]:
        """{spot: warp reference -> photo}; {} without a table (callers use IDENTITY)."""
        row = self._row(blob)
        if row is None:
            return {}
        return {s: tuple(float(row[f"{s}_{m}"]) for m in WARP_KEYS) for s in self.spots}


def align_image(image: Image.Image, warp: Warp) -> Image.Image:
    """Photo straightened to the geometry of the reference: output pixel (x, y) is taken from
    warp(x, y) in the photo. With the identity the photo is returned unchanged."""
    if tuple(warp) == IDENTITY:
        return image
    return image.transform(image.size, Image.AFFINE, data=tuple(warp), resample=Image.BILINEAR)


def transform_points(points_pct: list[list[float]], warp: Warp, size: tuple[int, int]) -> list[list[float]]:
    """Points of a polygon (in % of the reference) moved to where they fall in the photo, in %.

    Used to draw the ROIs in Label Studio on the original (not straightened) photo."""
    w, h = size
    m00, m01, m02, m10, m11, m12 = warp
    out = []
    for x, y in points_pct:
        px, py = x / 100 * w, y / 100 * h
        out.append([round((m00 * px + m01 * py + m02) / w * 100, 3), round((m10 * px + m11 * py + m12) / h * 100, 3)])
    return out


def write_csv(path: Path, rows: list[dict], landmarks, spots) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=BASE_COLUMNS + landmark_columns(landmarks) + spot_columns(spots))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: r["blob"]))
