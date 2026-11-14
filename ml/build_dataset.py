"""ParkWatch — builds the per-spot crop dataset for ParkNet.

  check_dataset export (export.json + check.json) + photos from Blob + rois.json
    -> <out>/crops/<photo>_<spot>.jpg   crop with a generous margin (crops.STORE_MARGIN)
    -> <out>/manifest.csv               one row per crop (see COLUMNS)
    -> <out>/dataset_info.json          fingerprint, ROI version, counts, skipped items

Rules:
  - From each task, the latest non-cancelled annotation; `exclude` is left out.
  - Geometry ALWAYS from rois.json; only the label is read from the annotation.
  - Every labelled photo goes in (also dark/washed-out ones); brightness and contrast of the
    evaluation crop (margin --quality-margin) are measured so they can be analysed apart.
  - Every photo is used; the episode of each crop is computed (consecutive run of photos of
    a spot with the same label, per day; `exclude` does not break it).
  - No splits here: train/validation/holdout are decided by day in train.py.

Meant to run as an Azure ML job (cluster identity) or locally (az login). The photo
Storage is read with the azure_utils credential, without keys.

Usage:
  python build_dataset.py --rois ../label_studio/rois.json --export-dir ../data/check --out ../data/parknet_ds
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from azure.storage.blob import BlobServiceClient
from PIL import Image

from azure_utils import credential, retry
from crops import STORE_MARGIN, clip_box, expand_box, polygon_bbox
from alignment import IDENTITY, Alignment, align_image, describe
from ls_client import latest_labels

CLASSES = ("free", "occupied")
NAME_RE = re.compile(r"parkwatch_(\d{8}T\d{6}Z)\.jpg$")
COLUMNS = [
    "crop", "blob", "day", "ts_utc", "spot", "label",
    "bx0", "by0", "bx1", "by1",  # polygon bbox inside the stored crop (px)
    "brightness", "contrast",  # mean and std of grey level (0-255) of the evaluation crop
    "episode_id", "episode_len", "episode_min",
    "rois_version",
    "align_dx", "align_dy", "align_angle",  # drift correction applied to this spot (alignment.py)
]


# --- Images ---------------------------------------------------------------------
def cached_image(container, blob: str, cache_dir: Path) -> Image.Image:
    path = cache_dir / blob
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(retry(lambda: container.download_blob(blob).readall(), blob))
    return Image.open(path).convert("RGB")


def gray_stats(image: Image.Image) -> tuple[float, float]:
    g = np.asarray(image.convert("L"), dtype=np.float32)
    return float(g.mean()), float(g.std())


# --- Episodes -------------------------------------------------------------------
def assign_episodes(rows: list[dict], all_labels: dict[tuple[str, str], list[tuple[str, str]]]) -> None:
    """Fills episode_id/len/min. `all_labels[(spot, day)]` = [(ts, label)], exclude included."""
    episode_of: dict[tuple[str, str], str] = {}
    sizes: Counter = Counter()
    span: dict[str, list[str]] = {}
    for (spot, day), seq in all_labels.items():
        k, prev = 0, None
        for ts, label in sorted(seq):
            if label not in CLASSES:  # exclude: does not break the episode
                continue
            if label != prev:
                k += 1
                prev = label
            eid = f"{day}_{spot}_{k:03d}"
            episode_of[(spot, ts)] = eid
            sizes[eid] += 1
            span.setdefault(eid, [ts, ts])[1] = ts
    for r in rows:
        eid = episode_of[(r["spot"], r["ts_utc"])]
        t0, t1 = (datetime.fromisoformat(t) for t in span[eid])
        r["episode_id"], r["episode_len"] = eid, sizes[eid]
        r["episode_min"] = round((t1 - t0).total_seconds() / 60, 1)


# --- Main -----------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rois", type=Path, required=True)
    p.add_argument("--export-dir", type=Path, required=True, help="check_dataset output (export.json, check.json)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cache", type=Path, default=None, help="cache of original photos (default <out>/../raw_cache)")
    p.add_argument("--storage-account", default="parkwatch")
    p.add_argument("--container", default="raw")
    p.add_argument("--alignment", type=Path, default=None, help="align_photos output (alignment.csv)")
    p.add_argument("--quality-margin", type=float, default=0.10, help="margin of the crop where brightness/contrast are measured")
    args = p.parse_args()
    cache = args.cache or args.out.parent / "raw_cache"

    rois = json.loads(args.rois.read_text(encoding="utf-8"))
    spot_ids = {s["id"] for s in rois["spots"]}
    tasks = json.loads((args.export_dir / "export.json").read_text(encoding="utf-8"))
    check = json.loads((args.export_dir / "check.json").read_text(encoding="utf-8"))
    alignment = Alignment.from_dir(args.alignment)
    container = BlobServiceClient(
        f"https://{args.storage_account}.blob.core.windows.net", credential=credential()
    ).get_container_client(args.container)

    # Rebuilt from scratch: the export is the source of truth. The contents are emptied, not the
    # folder itself, because in Azure ML <out> is the mount point of the job output.
    args.out.mkdir(parents=True, exist_ok=True)
    for child in args.out.iterdir():
        shutil.rmtree(child) if child.is_dir() else child.unlink()
    (args.out / "crops").mkdir()

    rows, skipped = [], Counter()
    all_labels: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for i, task in enumerate(tasks, 1):
        if i % 250 == 0:
            print(f"  {i}/{len(tasks)} tasks")
        blob = task["data"].get("blob") or ""
        m = NAME_RE.search(blob)
        if not m:
            skipped["no_blob"] += 1
            continue
        labels = latest_labels(task, spot_ids)
        if labels is None:
            skipped["no_annotation"] += 1
            continue
        ts = datetime.strptime(m.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        ts_iso, day = ts.isoformat(), ts.date().isoformat()
        for spot, label in labels.items():
            all_labels[(spot, day)].append((ts_iso, label))
        if not any(lbl in CLASSES for lbl in labels.values()):
            skipped["all_exclude"] += 1
            continue

        photo = cached_image(container, blob, cache)
        warps = alignment.warps(blob)  # one per spot: the fisheye moves every part of the image differently
        W, H = photo.size
        for spot in rois["spots"]:
            label = labels.get(spot["id"])
            if label not in CLASSES:
                skipped[f"spot_{label or 'no_region'}"] += 1
                continue
            warp = warps.get(spot["id"], IDENTITY)
            image = align_image(photo, warp)  # reference geometry of this spot
            drift = describe(warp)
            bbox = polygon_bbox(spot["points"], W, H)
            sx0, sy0, sx1, sy1 = clip_box(expand_box(bbox, STORE_MARGIN), W, H)
            stored = image.crop((sx0, sy0, sx1, sy1))
            quality = image.crop(clip_box(expand_box(bbox, args.quality_margin), W, H))
            brightness, contrast = gray_stats(quality)
            name = f"{Path(blob).stem}_{spot['id']}.jpg"
            stored.save(args.out / "crops" / name, quality=95)
            rows.append(
                {
                    "crop": f"crops/{name}",
                    "blob": blob,
                    "day": day,
                    "ts_utc": ts_iso,
                    "spot": spot["id"],
                    "label": label,
                    "bx0": round(bbox[0] - sx0, 2),
                    "by0": round(bbox[1] - sy0, 2),
                    "bx1": round(bbox[2] - sx0, 2),
                    "by1": round(bbox[3] - sy0, 2),
                    "brightness": round(brightness, 2),
                    "contrast": round(contrast, 2),
                    "rois_version": rois["version"],
                    "align_dx": round(drift["dx"], 1),
                    "align_dy": round(drift["dy"], 1),
                    "align_angle": round(drift["angle"], 3),
                }
            )

    assign_episodes(rows, all_labels)
    with open(args.out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    counts = defaultdict(Counter)
    for r in rows:
        counts[r["spot"]][r["label"]] += 1
    info = {
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fingerprint": check["fingerprint"],  # read by check_dataset on the next run
        "rois_version": rois["version"],
        "store_margin": STORE_MARGIN,
        "aligned": bool(alignment.rows),
        "quality_margin": args.quality_margin,
        "classes": list(CLASSES),
        "days": sorted({r["day"] for r in rows}),
        "n_crops": len(rows),
        "n_episodes": len({r["episode_id"] for r in rows}),
        "counts": {s: dict(c) for s, c in sorted(counts.items())},
        "skipped": dict(skipped),
    }
    (args.out / "dataset_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"\n{len(rows)} crops, {info['n_episodes']} episodes, days {info['days'][0]}..{info['days'][-1]}")
    print(f"  per spot: {info['counts']}")
    print(f"  skipped: {info['skipped']}")


if __name__ == "__main__":
    main()
