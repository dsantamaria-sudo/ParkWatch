"""ParkWatch — measures the camera drift photo by photo (first step of both pipelines).

  latest version of the table (data asset parkwatch-alignment) + new Blob photos
    -> <out>/alignment.csv   one row per photo: affine transform reference -> photo + inliers
    -> <out>/drift.json      smoothed drift of the last day + alert if it exceeds the thresholds

Each photo is measured only once (the table grows every night). Measurement: SIFT + RANSAC on
the static regions (static_mask.json) of the rois.json reference photo, with equalised contrast
(CLAHE): robust to lighting changes. See tools/camera_drift.py for the analysis behind it.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import statistics
from datetime import date, datetime, timedelta, timezone
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from alignment import COLUMNS, MIN_INLIERS, Alignment, blob_time, describe, write_csv
from azure_utils import credential, latest_asset_file, retry

SEARCH_MARGIN = 80  # px: points are searched in the photo within the dilated mask (the camera moves)
_state: dict = {}   # per Pool process: container, reference and mask


def static_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), np.uint8)
    for region in json.loads(path.read_text(encoding="utf-8"))["regions"]:
        cv2.fillPoly(mask, [np.array([[x / 100 * w, y / 100 * h] for x, y in region], np.int32)], 255)
    return mask


def gray(data: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    return cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(image)


def measure(reference: np.ndarray, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray | None, int]:
    """2x3 matrix reference -> photo (translation + rotation + scale) and matching points."""
    sift = cv2.SIFT_create(nfeatures=4000)
    kp_ref, des_ref = sift.detectAndCompute(reference, mask)
    kp_img, des_img = sift.detectAndCompute(image, cv2.dilate(mask, np.ones((2 * SEARCH_MARGIN + 1,) * 2, np.uint8)))
    if des_ref is None or des_img is None:
        return None, 0
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des_ref, des_img, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 8:
        return None, len(good)
    src = np.float32([kp_ref[m.queryIdx].pt for m in good])
    dst = np.float32([kp_img[m.trainIdx].pt for m in good])
    warp, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    return (warp, int(inliers.sum())) if warp is not None else (None, 0)


def _init(account: str, container: str, reference_blob: str, mask_path: str) -> None:
    from azure.storage.blob import BlobServiceClient

    client = BlobServiceClient(f"https://{account}.blob.core.windows.net", credential=credential())
    _state["container"] = client.get_container_client(container)
    _state["reference"] = gray(retry(lambda: _state["container"].download_blob(reference_blob).readall(), reference_blob))
    _state["mask"] = static_mask(Path(mask_path), _state["reference"].shape)


def _measure_blob(blob: str) -> dict:
    data = retry(lambda: _state["container"].download_blob(blob).readall(), blob)
    try:
        warp, inliers = measure(_state["reference"], gray(data), _state["mask"])
    except cv2.error:
        warp, inliers = None, 0
    m = warp.flatten().tolist() if warp is not None else [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    return {"blob": blob, "ts_utc": blob_time(blob).isoformat() + "Z",
            **{c: round(v, 6) for c, v in zip(COLUMNS[2:8], m)}, "inliers": inliers}


def list_photos(container, start: date, end: date) -> list[str]:
    blobs, day = [], start
    while day <= end:
        blobs += retry(lambda: [b.name for b in container.list_blobs(name_starts_with=f"{day:%Y/%m/%d}/")
                                if b.name.endswith(".jpg")], f"list {day}")
        day += timedelta(days=1)
    return sorted(blobs)


def drift_summary(alignment: Alignment, day: date, max_shift: float, max_angle: float, min_reliable: float) -> dict:
    blobs = [b for b in alignment.rows if blob_time(b).date() == day]
    if not blobs:
        return {"day": day.isoformat(), "n_photos": 0, "alert": False, "reasons": []}
    smoothed = [describe(alignment.warp(b)) for b in blobs]
    reliable = sum(int(alignment.rows[b]["inliers"]) >= MIN_INLIERS for b in blobs) / len(blobs)
    summary = {
        "day": day.isoformat(),
        "n_photos": len(blobs),
        "reliable_fraction": round(reliable, 3),
        "dx": round(statistics.median(s["dx"] for s in smoothed), 1),
        "dy": round(statistics.median(s["dy"] for s in smoothed), 1),
        "angle": round(statistics.median(s["angle"] for s in smoothed), 3),
        "max_shift": round(max(np.hypot(s["dx"], s["dy"]) for s in smoothed), 1),
        "max_angle": round(max(abs(s["angle"]) for s in smoothed), 3),
    }
    reasons = []
    if summary["max_shift"] > max_shift:
        reasons.append(f"shift {summary['max_shift']} px > {max_shift}")
    if summary["max_angle"] > max_angle:
        reasons.append(f"rotation {summary['max_angle']} degrees > {max_angle}")
    if reliable < min_reliable:
        reasons.append(f"only {reliable:.0%} of photos with a reliable measurement < {min_reliable:.0%}")
    return {**summary, "alert": bool(reasons), "reasons": reasons}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rois", type=Path, required=True)
    p.add_argument("--mask", type=Path, required=True, help="static_mask.json")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--asset-name", default="parkwatch-alignment")
    p.add_argument("--start", type=date.fromisoformat, default=date(2026, 10, 16), help="first day with photos")
    p.add_argument("--storage-account", default="parkwatch")
    p.add_argument("--container", default="raw")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    p.add_argument("--max-shift", type=float, default=40.0, help="alert: shift in px")
    p.add_argument("--max-angle", type=float, default=2.0, help="alert: rotation in degrees")
    p.add_argument("--min-reliable", type=float, default=0.5, help="alert: minimum fraction of measurable photos")
    args = p.parse_args()

    rois = json.loads(args.rois.read_text(encoding="utf-8"))
    mask_info = json.loads(args.mask.read_text(encoding="utf-8"))
    # same photo by name (parkwatch_<ts>Z.jpg): the path may be the Blob one or a Label Studio upload
    if Path(mask_info["source_image"]).name.split("-")[-1] != Path(rois["source_image"]).name:
        raise SystemExit(f"static_mask.json ({mask_info['source_image']}) and rois.json ({rois['source_image']}) "
                         "must use the same reference photo")

    previous = latest_asset_file(args.asset_name, "alignment.csv")
    rows: list[dict] = []
    if previous is not None:
        content, version = previous
        rows = list(csv.DictReader(io.StringIO(content.decode("utf-8"))))
        print(f"Previous table {args.asset_name}:{version}: {len(rows)} photos")

    yesterday = datetime.now(timezone.utc).date() - timedelta(days=1)
    _init(args.storage_account, args.container, rois["source_image"], str(args.mask))
    known = {r["blob"] for r in rows}
    new = [b for b in list_photos(_state["container"], args.start, yesterday) if b not in known]
    print(f"{len(new)} new photos to measure (up to {yesterday}), {args.workers} processes")
    with Pool(args.workers, initializer=_init,
              initargs=(args.storage_account, args.container, rois["source_image"], str(args.mask))) as pool:
        for i, row in enumerate(pool.imap_unordered(_measure_blob, new, chunksize=8), 1):
            rows.append(row)
            if i % 250 == 0:
                print(f"  {i}/{len(new)}")

    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "alignment.csv", rows)
    alignment = Alignment(rows)
    summary = drift_summary(alignment, yesterday, args.max_shift, args.max_angle, args.min_reliable)
    summary.update({"n_total": len(rows), "n_new": len(new)})
    (args.out / "drift.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Drift {summary['day']}: {json.dumps(summary)}")

    import mlflow

    with mlflow.start_run():
        mlflow.log_metrics({f"drift/{k}": float(v) for k, v in summary.items()
                            if isinstance(v, (int, float)) and not isinstance(v, bool)})
        mlflow.log_param("drift_alert", summary["alert"])


if __name__ == "__main__":
    main()
