"""ParkWatch — measures how much the camera has moved relative to the ROI reference photo.

For each day it takes the best photo of the window starting at --local-time (Madrid time, good
light) and estimates the transform relative to the rois.json reference photo. Default method:
SIFT + RANSAC (robust to lighting); --method ecc uses intensities (OpenCV ECC, euclidean model).
With --mask (static_mask.json) only the static background regions are used: no sky, sun, glass
reflections or cars. The `inliers` (or `cc`) column is the quality of the fit.

Outputs in --out (default data/drift/, outside git):
  drift.csv      blob, day, dx, dy (px at 1600x1200), rotation (degrees), scale, quality
  drift.png      dx, dy and rotation per photo over time
  overlay.png    latest photo with the rois.json ROIs (green) and the shifted ones (red)

Usage (repo root; reads the Storage with the connection string in tools/.env):
  uv run python tools/camera_drift.py --mask label_studio/static_mask.json
  uv run python tools/camera_drift.py --local-time 11:00
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

NAME_RE = re.compile(r"parkwatch_(\d{8}T\d{6}Z)\.jpg$")
SCALE = 0.5  # ECC at 800x600: faster and less sensitive to noise
MADRID = ZoneInfo("Europe/Madrid")


def load_gray(container, blob: str) -> np.ndarray:
    data = np.frombuffer(container.download_blob(blob).readall(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def prepare(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(small, (5, 5), 0).astype(np.float32)


def estimate(reference: np.ndarray, image: np.ndarray, mask: np.ndarray | None) -> tuple[np.ndarray, float]:
    """2x3 matrix (reference coordinates -> photo) at full scale and correlation coefficient."""
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
    cc, warp = cv2.findTransformECC(reference, image, warp, cv2.MOTION_EUCLIDEAN, criteria, mask, 5)
    warp[:, 2] /= SCALE
    return warp, float(cc)


def estimate_features(reference: np.ndarray, image: np.ndarray, mask: np.ndarray | None,
                      search_margin: int = 80) -> tuple[np.ndarray, int]:
    """SIFT + RANSAC on the full image (8-bit grey): 2x3 matrix reference -> photo
    (translation + rotation + scale) and number of matching points (inliers).

    Robust to lighting: a window corner stays in place even if the shadows change.
    In the photo, points are searched within the dilated mask (the camera may have moved)."""
    sift = cv2.SIFT_create(nfeatures=4000)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))  # local contrast: fog, overcast sky
    reference, image = clahe.apply(reference), clahe.apply(image)
    image_mask = cv2.dilate(mask, np.ones((2 * search_margin + 1,) * 2, np.uint8)) if mask is not None else None
    kp_ref, des_ref = sift.detectAndCompute(reference, mask)
    kp_img, des_img = sift.detectAndCompute(image, image_mask)
    if des_ref is None or des_img is None:
        raise ValueError("no keypoints")
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des_ref, des_img, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]  # Lowe's ratio
    if len(good) < 8:
        raise ValueError(f"only {len(good)} matches")
    src = np.float32([kp_ref[m.queryIdx].pt for m in good])
    dst = np.float32([kp_img[m.trainIdx].pt for m in good])
    warp, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if warp is None:
        raise ValueError("RANSAC found no solution")
    return warp.astype(np.float32), int(inliers.sum())


def static_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    """8-bit mask (at the given resolution) with the static regions of static_mask.json."""
    h, w = shape
    mask = np.zeros((h, w), np.uint8)
    for region in json.loads(path.read_text(encoding="utf-8"))["regions"]:
        pts = np.array([[x / 100 * w, y / 100 * h] for x, y in region], np.int32)
        cv2.fillPoly(mask, [pts], 255)
    return mask


def pick(blobs: list[str], day: date, local: time, window_min: int, n: int) -> list[str]:
    """Up to n photos spread over [local, local + window_min] (Madrid time) of that day."""
    start = datetime.combine(day, local, MADRID).astimezone(timezone.utc).replace(tzinfo=None)
    end = start + timedelta(minutes=window_min)
    inside = [b for b in blobs if NAME_RE.search(b)
              and start <= datetime.strptime(NAME_RE.search(b).group(1), "%Y%m%dT%H%M%SZ") <= end]
    if len(inside) <= n:
        return inside
    step = (len(inside) - 1) / (n - 1)
    return [inside[round(i * step)] for i in range(n)]


def draw_rois(image: np.ndarray, rois: dict, warp: np.ndarray | None, color) -> None:
    h, w = image.shape[:2]
    for spot in rois["spots"]:
        pts = np.array([[x / 100 * w, y / 100 * h] for x, y in spot["points"]], np.float32)
        if warp is not None:
            pts = cv2.transform(pts[None], warp)[0]
        cv2.polylines(image, [pts.astype(np.int32)], True, color, 2)
        cx, cy = pts.mean(axis=0).astype(int)
        cv2.putText(image, spot["id"], (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rois", type=Path, default=Path("label_studio/rois.json"))
    p.add_argument("--local-time", type=time.fromisoformat, default=time(10, 0), help="start of the window (Madrid time)")
    p.add_argument("--window-min", type=int, default=60, help="window length in minutes")
    p.add_argument("--candidates", type=int, default=5,
                   help="photos tried per day within the window; keeps the one with most matches (sift)")
    p.add_argument("--mask", type=Path, default=None, help="static_mask.json: measure only in the static regions")
    p.add_argument("--method", choices=["sift", "ecc"], default="sift",
                   help="sift: keypoints + RANSAC (robust to lighting); ecc: intensities")
    p.add_argument("--start", type=date.fromisoformat, default=date(2026, 10, 16))
    p.add_argument("--out", type=Path, default=Path("data/drift"))
    args = p.parse_args()

    load_dotenv(Path(__file__).with_name(".env"))
    container = BlobServiceClient.from_connection_string(
        os.environ["AZURE_STORAGE_CONNECTION_STRING"]
    ).get_container_client(os.getenv("PW_CONTAINER", "raw"))
    rois = json.loads(args.rois.read_text(encoding="utf-8"))
    ref_image = load_gray(container, rois["source_image"])
    if args.method == "ecc":
        reference = prepare(ref_image)
    else:
        reference = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
    mask = static_mask(args.mask, reference.shape) if args.mask else None
    quality = "cc" if args.method == "ecc" else "inliers"
    print(f"Reference: {rois['source_image']} (rois {rois['version']}), "
          f"photo at {args.local_time:%H:%M} Madrid time, method {args.method}, "
          + (f"mask {args.mask} ({(mask > 0).mean():.0%} of the image)" if mask is not None else "no mask"))

    args.out.mkdir(parents=True, exist_ok=True)
    rows, last = [], None
    today = datetime.now(timezone.utc).date()
    day = args.start
    while day < today:
        blobs = sorted(b.name for b in container.list_blobs(name_starts_with=f"{day:%Y/%m/%d}/") if b.name.endswith(".jpg"))
        best = None  # (quality, blob, image, warp)
        for blob in pick(blobs, day, args.local_time, args.window_min, args.candidates):
            image = load_gray(container, blob)
            try:
                if args.method == "ecc":
                    warp, q = estimate(reference, prepare(image), mask)
                else:
                    warp, q = estimate_features(reference, cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), mask)
            except (cv2.error, ValueError) as exc:
                print(f"  {blob}: no estimate ({exc})")
                continue
            if best is None or q > best[0]:
                best = (q, blob, image, warp)
        if best is not None:
            q, blob, image, warp = best
            angle = float(np.degrees(np.arctan2(warp[1, 0], warp[0, 0])))
            scale = float(np.hypot(warp[0, 0], warp[1, 0]))
            rows.append({"blob": blob, "day": day.isoformat(), "dx": round(float(warp[0, 2]), 1),
                         "dy": round(float(warp[1, 2]), 1), "angle": round(angle, 3), "scale": round(scale, 4),
                         quality: round(q, 3)})
            last = (blob, image, warp)
            print(f"  {blob}: dx {warp[0, 2]:+6.1f} px  dy {warp[1, 2]:+6.1f} px  rotation {angle:+.3f}  "
                  f"scale {scale:.4f}  {quality} {q:.3f}" if args.method == "ecc" else
                  f"  {blob}: dx {warp[0, 2]:+6.1f} px  dy {warp[1, 2]:+6.1f} px  rotation {angle:+.3f}  "
                  f"scale {scale:.4f}  {quality} {q}")
        day += timedelta(days=1)

    with open(args.out / "drift.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    xs = range(len(rows))
    axes[0].plot(xs, [r["dx"] for r in rows], marker="o", label="dx (px)")
    axes[0].plot(xs, [r["dy"] for r in rows], marker="o", label="dy (px)")
    axes[0].axhline(0, color="gray", lw=0.8)
    axes[0].set_ylabel("shift (px, 1600x1200)")
    axes[0].legend()
    axes[1].plot(xs, [r["angle"] for r in rows], marker="o", color="tab:red")
    axes[1].axhline(0, color="gray", lw=0.8)
    axes[1].set_ylabel("rotation (degrees)")
    ticks = [i for i, r in enumerate(rows) if i == 0 or r["day"] != rows[i - 1]["day"]]
    axes[1].set_xticks(ticks, [rows[i]["day"][5:] for i in ticks], rotation=45)
    fig.suptitle(f"Camera drift relative to {rois['source_image']}")
    fig.tight_layout()
    fig.savefig(args.out / "drift.png", dpi=110)

    blob, image, warp = last
    draw_rois(image, rois, None, (0, 200, 0))
    draw_rois(image, rois, warp, (0, 0, 255))
    cv2.putText(image, f"{blob}  green: rois.json  red: shifted", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.imwrite(str(args.out / "overlay.png"), image)
    print(f"\n{len(rows)} photos -> {args.out / 'drift.csv'}, drift.png, overlay.png")


if __name__ == "__main__":
    main()
