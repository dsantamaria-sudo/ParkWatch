"""ParkWatch — alignment of every photo with the reference photo of the ROIs (camera drift).

The camera is glued in place and moves over time (tools/camera_drift.py). align_photos.py
measures, for every photo, the affine transform reference -> photo (SIFT + RANSAC on the
static regions) and stores it in a table (alignment.csv). This module uses that table:

  - warp(blob): transform smoothed over time (median of the reliable measurements within
    +-WINDOW_MIN of the photo; otherwise the nearest reliable one; with none at all, identity).
  - align_image(): straightens the photo to the geometry of the reference, so rois.json
    applies as is. Same step in build_dataset and in inference.

No OpenCV: only numpy + PIL (the measurement, with OpenCV, lives in align_photos.py).
"""
from __future__ import annotations

import bisect
import csv
import math
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from PIL import Image

COLUMNS = ["blob", "ts_utc", "m00", "m01", "m02", "m10", "m11", "m12", "inliers"]
MIN_INLIERS = 15   # below this the measurement is not reliable (19 already aligns well: preview of 2/11)
WINDOW_MIN = 60    # smoothing: reliable measurements within +-60 min of the photo
IDENTITY = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
NAME_RE = re.compile(r"parkwatch_(\d{8}T\d{6}Z)\.jpg$")

Warp = tuple[float, float, float, float, float, float]  # m00 m01 m02 / m10 m11 m12 (reference -> photo)


def blob_time(blob: str) -> datetime:
    return datetime.strptime(NAME_RE.search(blob).group(1), "%Y%m%dT%H%M%SZ")


def describe(warp: Warp) -> dict:
    """Shift (px), rotation (degrees) and scale of a transform."""
    m00, _, m02, m10, _, m12 = warp
    return {"dx": m02, "dy": m12, "angle": math.degrees(math.atan2(m10, m00)), "scale": math.hypot(m00, m10)}


class Alignment:
    def __init__(self, rows: list[dict]):
        self.rows = {r["blob"]: r for r in rows}
        reliable = sorted(
            (blob_time(r["blob"]), tuple(float(r[c]) for c in COLUMNS[2:8]))
            for r in rows if int(r["inliers"]) >= MIN_INLIERS
        )
        self._times = [t for t, _ in reliable]
        self._warps = [w for _, w in reliable]

    @classmethod
    def from_dir(cls, path: Path | None) -> "Alignment":
        """Table from a folder (align_photos output); without folder or table, empty (identity)."""
        csv_path = Path(path) / "alignment.csv" if path else None
        if csv_path is None or not csv_path.exists():
            return cls([])
        with open(csv_path, newline="", encoding="utf-8") as f:
            return cls(list(csv.DictReader(f)))

    def warp(self, blob: str) -> Warp:
        if not self._times:
            return IDENTITY
        t = blob_time(blob)
        window = timedelta(minutes=WINDOW_MIN)
        lo = bisect.bisect_left(self._times, t - window)
        hi = bisect.bisect_right(self._times, t + window)
        if hi > lo:
            return tuple(float(v) for v in np.median(np.array(self._warps[lo:hi]), axis=0))
        i = bisect.bisect_left(self._times, t)
        nearest = min((j for j in (i - 1, i) if 0 <= j < len(self._times)), key=lambda j: abs(self._times[j] - t))
        return self._warps[nearest]


def align_image(image: Image.Image, warp: Warp) -> Image.Image:
    """Photo straightened to the geometry of the reference: output pixel (x, y) is taken from
    warp(x, y) in the photo. With the identity the photo is returned unchanged."""
    if warp == IDENTITY:
        return image
    return image.transform(image.size, Image.AFFINE, data=warp, resample=Image.BILINEAR)


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


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: r["blob"]))
