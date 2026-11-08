"""ParkWatch — landmarks and anchors from the Label Studio landmarks project -> label_studio/landmarks.json.

  - reference photo (the one of rois.json): position of every landmark (lm1..lmN)
  - any other photo: an ANCHOR, i.e. the exact camera position for that photo.

For every anchor it prints the shift of every landmark, a rigid fit (rotation + shift) and the
leave-one-out check of the local model the pipeline uses (ml/landmarks.py): a landmark the
others predict badly is a misplaced click (beyond OUTLIER_PX it is dropped, as the pipeline
does). Then the shift at the centre of every spot, next to the distance to the nearest landmark
(far from any landmark the value is an extrapolation). With --compare, the same from the
current alignment table (format 2, written by align_photos).

The pipeline reads the anchors straight from Label Studio; landmarks.json is for the local
tools (tools/landmark_tracking.py) and as a record. Also draws label_studio/landmarks.png
(reference with the numbered landmarks and the spots; not in git) to know which point is which
when clicking anchors.

Usage (repo root; LS_URL, LS_TOKEN and AZURE_STORAGE_CONNECTION_STRING in tools/.env):
  uv run python tools/landmarks_from_ls.py --project 7
  uv run python tools/landmarks_from_ls.py --project 7 --compare   # vs the current alignment table (az login)
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml"))
from alignment import Alignment, describe  # noqa: E402
from landmarks import (  # noqa: E402
    apply, drop_outliers, fit_rigid, keypoints, leave_one_out, local_warp, spot_centres, task_blob, to_px,
)
from ls_client import export_tasks  # noqa: E402

RESIDUAL_WARN_PX = 4.0  # rigid fit: a landmark further than this is worth a look (fisheye: often fine)


def load_table() -> Alignment | None:
    os.environ.setdefault("AZUREML_ARM_SUBSCRIPTION", "48ee8f92-9a0c-4ecd-b6ce-83845f85bf75")
    os.environ.setdefault("AZUREML_ARM_RESOURCEGROUP", "parkwatch")
    os.environ.setdefault("AZUREML_ARM_WORKSPACE_NAME", "parkwatch-workspace")
    from azure_utils import latest_asset_file

    content, version = latest_asset_file("parkwatch-alignment", "alignment.csv")
    rows = list(csv.DictReader(io.StringIO(content.decode("utf-8"))))
    if rows and "status" not in rows[0]:
        print(f"parkwatch-alignment:{version} is the old table (one warp per photo): no comparison")
        return None
    print(f"Compared with parkwatch-alignment:{version}")
    return Alignment(rows)


def draw_reference(blob: str, landmarks: dict[str, list[float]], rois: dict, out: Path) -> None:
    from azure.storage.blob import BlobServiceClient
    from PIL import Image, ImageDraw

    container = BlobServiceClient.from_connection_string(os.environ["AZURE_STORAGE_CONNECTION_STRING"]) \
        .get_container_client(os.getenv("PW_CONTAINER", "raw"))
    image = Image.open(io.BytesIO(container.download_blob(blob).readall())).convert("RGB")
    d = ImageDraw.Draw(image)
    w, h = image.size
    for spot in rois["spots"]:  # spots in blue: every spot needs landmarks around it
        pts = [(x / 100 * w, y / 100 * h) for x, y in spot["points"]]
        d.line(pts + [pts[0]], fill="deepskyblue", width=3)
        d.text((sum(x for x, _ in pts) / len(pts) - 15, sum(y for _, y in pts) / len(pts) - 15),
               spot["id"], fill="deepskyblue", font_size=28)
    for label, (x, y) in sorted(landmarks.items()):
        px, py = x / 100 * w, y / 100 * h
        d.ellipse((px - 9, py - 9, px + 9, py + 9), outline="red", width=3)
        d.text((px + 12, py - 12), label, fill="red", font_size=28)
    image.save(out)
    print(f"Reference with the landmarks -> {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project", type=int, required=True, help="id of the parkwatch-landmarks project")
    p.add_argument("-o", "--out", type=Path, default=ROOT / "label_studio" / "landmarks.json")
    p.add_argument("--no-png", action="store_true", help="do not draw label_studio/landmarks.png")
    p.add_argument("--compare", action="store_true", help="compare every anchor with the current alignment table")
    args = p.parse_args()

    load_dotenv(ROOT / "tools" / ".env")
    rois = json.loads((ROOT / "label_studio" / "rois.json").read_text(encoding="utf-8"))
    reference = rois["source_image"]
    ls_url = os.environ["LS_URL"].rstrip("/")
    marked = {}
    for task in export_tasks(ls_url, os.environ["LS_TOKEN"], args.project):
        blob, kp = task_blob(task), keypoints(task)
        if blob and kp:
            marked[blob] = kp
    if reference not in marked:
        raise SystemExit(f"The reference photo {reference} has no landmarks yet (task created with --reference?)")
    ref_pct, size = marked.pop(reference)
    ref = to_px(ref_pct, size)
    print(f"Reference {reference}: {len(ref)} landmarks ({', '.join(sorted(ref, key=lambda k: int(k[2:])))})")
    if len(ref) < 3:
        print("  warning: at least 3 landmarks are needed (spread over the image, some next to every spot)")
    table = load_table() if args.compare else None
    centres = spot_centres(rois, size)

    anchors = {}
    for blob, (pct, _) in sorted(marked.items()):
        points, dropped = drop_outliers(ref, to_px(pct, size))
        for k in dropped:
            print(f"  warning: {blob}: {k} does not fit the other landmarks -> dropped (fix or delete that click)")
        common = sorted(points, key=lambda k: int(k[2:]))
        if len(common) < 2:
            print(f"{blob}: only {len(common)} landmark(s) in common with the reference, skipped")
            continue
        src, dst = np.array([ref[k] for k in common]), np.array([points[k] for k in common])
        warp, residuals = fit_rigid(src, dst)
        anchors[blob] = {"landmarks": {k: [round(v, 4) for v in pct[k]] for k in common},
                         "warp": [round(v, 6) for v in warp]}
        shift = (dst - src).mean(axis=0)
        print(f"{blob}: {len(common)} landmarks, mean shift ({shift[0]:.0f}, {shift[1]:.0f}) px, "
              f"rotation {describe(warp)['angle']:.2f}°")
        print("    shift of every landmark: " + "  ".join(
            f"{k} ({x:.0f},{y:.0f})" for k, (x, y) in zip(common, dst - src)))
        if len(common) >= 4:
            rigid_err, local_err = leave_one_out(src, dst)
            print(f"    leave-one-out error: global rigid mean {rigid_err.mean():.0f} px (worst {rigid_err.max():.0f}) | "
                  f"local mean {local_err.mean():.0f} px (worst {local_err.max():.0f}, "
                  f"{common[int(local_err.argmax())]})")
        in_table = table is not None and blob in table.rows
        table_warps = table.warps(blob) if in_table else {}
        print(f"    {'spot':5} {'nearest lm':>11} {'local':>14} {'global rigid':>14}"
              + (f" {'table':>14} {'table err':>9}" if in_table else ""))
        for sid, c in centres.items():
            local = apply(local_warp(src, dst, c), c[None])[0] - c
            rigid = apply(warp, c[None])[0] - c
            near = np.hypot(*(src - c).T).min()
            line = f"    {sid:5} {near:8.0f} px ({local[0]:5.0f}, {local[1]:5.0f}) ({rigid[0]:5.0f}, {rigid[1]:5.0f})"
            if sid in table_warps:
                theirs = apply(table_warps[sid], c[None])[0] - c
                line += f" ({theirs[0]:5.0f}, {theirs[1]:5.0f}) {np.hypot(*(theirs - local)):6.0f} px"
            print(line)
        worst = max(zip(residuals, common))
        if worst[0] > RESIDUAL_WARN_PX * 5:
            print(f"    note: rigid residual {worst[0]:.0f} px at {worst[1]} (fisheye; the local model above is what counts)")

    out = {
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source_image": reference,
        "image_size": list(size),
        "landmarks": {k: [round(v, 4) for v in ref_pct[k]] for k in sorted(ref_pct, key=lambda k: int(k[2:]))},
        "anchors": anchors,
    }
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"{len(ref)} landmarks + {len(anchors)} anchors -> {args.out}")
    if not args.no_png:
        draw_reference(reference, ref_pct, rois, args.out.with_suffix(".png"))


if __name__ == "__main__":
    main()
