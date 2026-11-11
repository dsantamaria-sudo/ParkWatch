"""ParkWatch — camera drift, photo by photo (first step of both pipelines).

  anchors (Label Studio project parkwatch-landmarks) + latest table (data asset
  parkwatch-alignment) + new Blob photos
    -> <out>/alignment.csv        one row per photo: landmark positions + warp of every spot
    -> <out>/alignment_meta.json  table format and reference landmarks (a change rebuilds the table)
    -> <out>/drift.json           summary of the last day + alert (what to anchor)

The landmarks are tracked photo by photo (landmarks.py). Each night the tracking continues
from the last photo of the table. If an anchor appears on a photo that is already in the table
(clicked after an alert), the table is cut there and tracked again from the anchor. The warps
of every spot are recomputed for the whole table on every run (cheap; follows rois.json).

Alert (drift_alert fails, email): a jump of a spot of more than JUMP_PX between two photos
(the camera moved, or the tracking got confused), or a lost stretch in daylight longer than
LOST_ALERT photos. The message gives the photo to anchor:
  uv run python tools/landmark_tasks.py --project <id> --at "<UTC time>"
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from alignment import TABLE_FORMAT, WARP_KEYS, Alignment, blob_time, write_csv
from azure_utils import credential, get_secret, latest_asset_file, retry
from landmarks import Tracker, anchors_from_tasks, apply, spot_centres, spot_warps
from ls_client import export_tasks

TRACKING_VERSION = 2  # bump when landmarks.py changes how positions are measured: rebuilds the table
JUMP_PX = 15          # alert: a spot centre moving more than this between two photos
LOST_ALERT = 20       # alert: lost photos in a row in daylight (~1 h)


def list_photos(container, start: date, end: date) -> list[str]:
    blobs, day = [], start
    while day <= end:
        blobs += retry(lambda: [b.name for b in container.list_blobs(name_starts_with=f"{day:%Y/%m/%d}/")
                                if b.name.endswith(".jpg")], f"list {day}")
        day += timedelta(days=1)
    return sorted(blobs)


def positions_of(row: dict, landmarks) -> dict:
    return {k: np.array([float(row[f"{k}_x"]), float(row[f"{k}_y"])]) for k in landmarks if f"{k}_x" in row}


def resume_state(rows: list[dict], landmarks) -> tuple[dict, dict, bool]:
    """Positions, photos missed per landmark and lost flag at the end of the table."""
    last = rows[-1]
    missed = {k: 0 for k in landmarks}
    for k in landmarks:
        for r in reversed(rows[-10:]):
            if k in r["found"].split(";") or r["status"] == "anchor":
                break
            missed[k] += 1
    return positions_of(last, landmarks), missed, last["status"] == "lost"


def daylight_lost_stretches(rows: list[dict]) -> list[tuple[str, str, int]]:
    """Lost stretches of one day that touch neither its first nor its last photo (dark ends)."""
    out, start, n = [], None, 0
    for i, r in enumerate(rows):
        if r["status"] == "lost":
            start, n = (start or (i, r["blob"])), n + 1
        else:
            if start and start[0] > 0:
                out.append((start[1], rows[i - 1]["blob"], n))
            start, n = None, 0
    return out


def drift_summary(rows: list[dict], centres: dict, day: date, project: int) -> dict:
    day_rows = [r for r in rows if blob_time(r["blob"]).date() == day]
    if not day_rows:
        return {"day": day.isoformat(), "n_photos": 0, "alert": False, "reasons": []}
    shifts = []  # per photo: {spot: centre shift}
    for r in day_rows:
        warps = {s: tuple(float(r[f"{s}_{m}"]) for m in WARP_KEYS) for s in centres}
        shifts.append({s: apply(warps[s], centres[s][None])[0] - centres[s] for s in centres})
    jumps, prev = [], None
    for r, sh in zip(day_rows, shifts):
        if r["status"] == "lost":
            continue
        if prev is not None and r["status"] != "anchor":  # an anchor correcting the tracking is not a jump
            moved = max(float(np.hypot(*(sh[s] - prev[s]))) for s in centres)
            if moved > JUMP_PX:
                jumps.append((r["blob"], round(moved, 1)))
        prev = sh
    lost = [s for s in daylight_lost_stretches(day_rows) if s[2] >= LOST_ALERT]
    anchor_cmd = 'uv run python tools/landmark_tasks.py --project {p} --at "{t:%Y-%m-%d %H:%M}"'
    reasons = [f"jump of {px} px at {blob} -> check it / anchor it: {anchor_cmd.format(p=project, t=blob_time(blob))}"
               for blob, px in jumps[:3]]
    if len(jumps) > 3:
        reasons.append(f"... and {len(jumps) - 3} more jumps that day (anchoring the first ones usually fixes them)")
    reasons += [f"lost in daylight from {a} ({n} photos) -> anchor: {anchor_cmd.format(p=project, t=blob_time(a))}"
                for a, _, n in lost]
    n_lost = sum(r["status"] == "lost" for r in day_rows)
    median = {s: np.median(np.array([sh[s] for sh in shifts]), axis=0) for s in centres}
    return {
        "day": day.isoformat(),
        "n_photos": len(day_rows),
        "tracked_fraction": round(1 - n_lost / len(day_rows), 3),
        "max_jump_px": max((px for _, px in jumps), default=0.0),
        **{f"{s}_dx": round(float(v[0]), 1) for s, v in median.items()},
        **{f"{s}_dy": round(float(v[1]), 1) for s, v in median.items()},
        "alert": bool(reasons),
        "reasons": reasons,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rois", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--asset-name", default="parkwatch-alignment")
    p.add_argument("--ls-url", required=True)
    p.add_argument("--landmarks-project", type=int, required=True, help="Label Studio project parkwatch-landmarks")
    p.add_argument("--keyvault-url", required=True)
    p.add_argument("--secret-name", default="label-studio-token")
    p.add_argument("--rebuild", default="false", help="true = track the whole history again")
    p.add_argument("--start", type=date.fromisoformat, default=date(2026, 10, 16), help="first day with photos")
    p.add_argument("--storage-account", default="parkwatch")
    p.add_argument("--container", default="raw")
    p.add_argument("--workers", type=int, default=8, help="parallel downloads")
    args = p.parse_args()

    from azure.storage.blob import BlobServiceClient

    rois = json.loads(args.rois.read_text(encoding="utf-8"))
    token = os.getenv("LS_TOKEN") or get_secret(args.keyvault_url, args.secret_name)
    tasks = export_tasks(args.ls_url.rstrip("/"), token, args.landmarks_project)
    ref, anchors, size, warnings = anchors_from_tasks(tasks, rois["source_image"])
    for w in warnings:
        print(f"warning: {w}")
    landmarks = sorted(ref)
    print(f"{len(ref)} landmarks, {len(anchors)} anchors (reference included)")
    meta = {"table_format": TABLE_FORMAT, "tracking_version": TRACKING_VERSION,
            "reference": rois["source_image"], "image_size": list(size),
            "landmarks_hash": hashlib.sha256(json.dumps({k: [round(float(v), 1) for v in ref[k]] for k in landmarks},
                                                        sort_keys=True).encode()).hexdigest()[:16]}

    # previous table, unless the landmarks, the format or the tracking changed
    rows: list[dict] = []
    previous = None if args.rebuild.lower() == "true" else latest_asset_file(args.asset_name, "alignment_meta.json")
    if previous is not None:
        prev_meta = json.loads(previous[0].decode("utf-8"))
        same = all(prev_meta.get(k) == meta[k] for k in ("table_format", "tracking_version", "reference", "landmarks_hash"))
        table = latest_asset_file(args.asset_name, "alignment.csv") if same else None
        # the table must come from the same version as its metadata (an older pipeline may have
        # registered a newer version in format 1 meanwhile)
        if table is not None and table[1] == previous[1]:
            rows = sorted(csv.DictReader(io.StringIO(table[0].decode("utf-8"))), key=lambda r: r["blob"])
            if rows and "status" not in rows[0]:
                rows = []
        if rows:
            print(f"Previous table {args.asset_name}:{table[1]}: {len(rows)} photos")
        else:
            print("Landmarks, table format or tracking changed (or no usable table): the whole history is tracked again")
    else:
        print("No previous table in format 2 (or --rebuild): the whole history is tracked")

    # a new anchor on a photo already in the table: cut the table there and track again from it
    new_anchor = next((i for i, r in enumerate(rows) if r["blob"] in anchors and r["status"] != "anchor"), None)
    if new_anchor is not None:
        print(f"New anchor {rows[new_anchor]['blob']}: tracking again from it ({len(rows) - new_anchor} photos)")
        rows = rows[:new_anchor]

    container = BlobServiceClient(f"https://{args.storage_account}.blob.core.windows.net",
                                  credential=credential()).get_container_client(args.container)
    download = lambda blob: retry(lambda: container.download_blob(blob).readall(), blob)  # noqa: E731
    tracker = Tracker(ref, anchors, download)
    yesterday = datetime.now(timezone.utc).date() - timedelta(days=1)
    if rows:
        positions, missed, lost = resume_state(rows, landmarks)
        tracker.start(positions, missed, lost)
        start = blob_time(rows[-1]["blob"]).date()
    else:
        start = args.start
        tracker.start(near=f"x/parkwatch_{start:%Y%m%d}T120000Z.jpg")
    known = {r["blob"] for r in rows}
    last_time = blob_time(rows[-1]["blob"]) if rows else None
    new = [b for b in list_photos(container, start, yesterday)
           if b not in known and (last_time is None or blob_time(b) > last_time)]
    print(f"{len(new)} new photos to track (up to {yesterday})")

    centres = spot_centres(rois, size)
    with ThreadPoolExecutor(args.workers) as pool:  # downloads ahead, in order
        for i, (blob, data) in enumerate(zip(new, pool.map(download, new)), 1):
            result = tracker.step(blob, data)
            if result["before_anchor"] is not None:
                clicked = anchors[blob]
                err = np.mean([np.hypot(*(result["before_anchor"][k] - clicked[k])) for k in clicked])
                print(f"  anchor {blob}: the tracking was {err:.1f} px from the clicks")
            row = {"blob": blob, "ts_utc": blob_time(blob).isoformat() + "Z", "status": result["status"],
                   "n_found": len(result["found"]), "found": ";".join(result["found"])}
            for k in landmarks:
                row[f"{k}_x"], row[f"{k}_y"] = round(float(tracker.positions[k][0]), 2), round(float(tracker.positions[k][1]), 2)
            rows.append(row)
            if i % 500 == 0:
                print(f"  {i}/{len(new)}")

    # warps of every spot, whole table (follows the current rois.json)
    for r in rows:
        for s, warp in spot_warps(ref, positions_of(r, landmarks), centres).items():
            for m, v in zip(WARP_KEYS, warp):
                r[f"{s}_{m}"] = round(v, 6)

    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "alignment.csv", rows, landmarks, list(centres))
    (args.out / "alignment_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    summary = drift_summary(rows, centres, yesterday, args.landmarks_project)
    summary.update({"n_total": len(rows), "n_new": len(new), "n_anchors": len(anchors), "warnings": warnings})
    (args.out / "drift.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Drift {summary['day']}: {json.dumps(summary)}")
    Alignment(rows)  # sanity check: the table can be read back

    import mlflow

    with mlflow.start_run():
        mlflow.log_metrics({f"drift/{k}": float(v) for k, v in summary.items()
                            if isinstance(v, (int, float)) and not isinstance(v, bool)})
        mlflow.log_param("drift_alert", summary["alert"])


if __name__ == "__main__":
    main()
