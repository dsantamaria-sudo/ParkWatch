"""ParkWatch — local run of the landmark tracking (ml/landmarks.py) over a range of days.

The same tracking align_photos.py runs in the pipeline, here with the anchors of
label_studio/landmarks.json (tools/landmarks_from_ls.py) and the photos read with the
connection string, to analyse it: every day starts where the previous one ended, the
reference photo counts as an anchor, and on every anchor the tracking is first compared with
the clicks (validation) and then reset to them.

Output in data/landmarks/<day>/ (outside git): tracking.csv (one row per photo: status,
landmarks found, mean match score, shift of every spot) and, with --preview, every photo drawn
with the moved spots (green; red: current alignment table with --table) plus index.html to
browse them; with --video, the same frames for the whole range in a timelapse MP4
(data/landmarks/timelapse_<start>_<end>.mp4). Local only: the photos never leave the machine.

Usage (repo root; AZURE_STORAGE_CONNECTION_STRING in tools/.env):
  uv run python tools/landmark_tracking.py --start 2026-10-16 --end 2026-11-07
  uv run python tools/landmark_tracking.py --start 2026-11-06 --preview --table
  uv run python tools/landmark_tracking.py --start 2026-10-16 --end 2026-11-07 --video
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import cv2
import numpy as np
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml"))
from alignment import Alignment  # noqa: E402
from landmarks import Tracker, apply, local_warp, spot_centres, spot_warps, to_px  # noqa: E402


def load_table() -> Alignment | None:
    from azure_utils import latest_asset_file

    os.environ.setdefault("AZUREML_ARM_SUBSCRIPTION", "48ee8f92-9a0c-4ecd-b6ce-83845f85bf75")
    os.environ.setdefault("AZUREML_ARM_RESOURCEGROUP", "parkwatch")
    os.environ.setdefault("AZUREML_ARM_WORKSPACE_NAME", "parkwatch-workspace")
    content, version = latest_asset_file("parkwatch-alignment", "alignment.csv")
    rows = list(csv.DictReader(io.StringIO(content.decode("utf-8"))))
    if rows and "status" not in rows[0]:
        print(f"parkwatch-alignment:{version} is the old table (one warp per photo): not drawn")
        return None
    print(f"Current table: parkwatch-alignment:{version}")
    return Alignment(rows)


def spot_error(ref: dict, tracked: dict, clicked: dict, centres: dict) -> dict[str, float]:
    """Distance between the spot centres moved with the tracking and with the clicks."""
    a, b = spot_warps(ref, tracked, centres), spot_warps(ref, clicked, centres)
    return {s: float(np.hypot(*(apply(a[s], c[None])[0] - apply(b[s], c[None])[0]))) for s, c in centres.items()}


def draw(image, rois, ref, positions, found, size, table_warps, title, path, width: int = 960):
    """Spots moved with the local model (green) and with the current table (red); landmarks found
    (yellow) or carried with the common movement (grey)."""
    w, h = size
    src = np.array([ref[k] for k in positions])
    dst = np.array(list(positions.values()))
    for spot in rois["spots"]:
        pts = np.array([[x / 100 * w, y / 100 * h] for x, y in spot["points"]])
        if table_warps and spot["id"] in table_warps:
            cv2.polylines(image, [apply(table_warps[spot["id"]], pts).astype(np.int32)], True, (60, 60, 230), 2)
        moved = apply(local_warp(src, dst, pts.mean(axis=0)), pts)
        cv2.polylines(image, [moved.astype(np.int32)], True, (60, 220, 60), 3)
        cx, cy = moved.mean(axis=0).astype(int)
        cv2.putText(image, spot["id"], (cx - 20, cy + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (60, 220, 60), 2)
    for k, (x, y) in positions.items():
        if k in found:
            cv2.circle(image, (int(x), int(y)), 7, (0, 230, 255), -1)
        else:
            cv2.circle(image, (int(x), int(y)), 7, (160, 160, 160), 2)
    cv2.rectangle(image, (0, 0), (w, 50), (0, 0, 0), -1)
    cv2.putText(image, title, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    image = cv2.resize(image, (width, int(width * h / w)), interpolation=cv2.INTER_AREA)
    if path is not None:
        cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return image


def write_viewer(path: Path, day: str, frames: list[dict], with_table: bool) -> None:
    legend = ("<b style='color:#3cdc3c'>green</b>: spots with the landmarks"
              + (" &nbsp; <b style='color:#e63c3c'>red</b>: current alignment table" if with_table else "")
              + " &nbsp; <b style='color:#ffe600'>yellow</b>: landmark found &nbsp; <b style='color:#aaa'>grey</b>: not found (carried)")
    path.write_text(f"""<!doctype html><html><head><meta charset="utf-8"><title>Landmarks {day}</title>
<style>body{{background:#111;color:#ddd;font:14px sans-serif;margin:16px}}img{{max-width:100%;display:block;margin:8px 0}}
input[type=range]{{width:100%}}button{{margin-right:6px}}#info{{font-size:16px}}</style></head><body>
<div>{legend}</div>
<div><button id="prev">&larr;</button><button id="play">play</button><button id="next">&rarr;</button>
<span id="info"></span></div><input id="slider" type="range" min="0" value="0"><img id="img">
<script>
const frames = {json.dumps(frames)};
const s = document.getElementById('slider'), img = document.getElementById('img'), info = document.getElementById('info');
s.max = frames.length - 1;
function show(i) {{ i = Math.max(0, Math.min(frames.length - 1, i)); s.value = i; const f = frames[i];
  img.src = f.file; info.textContent = ` ${{i + 1}}/${{frames.length}}  ${{f.time}} UTC  ${{f.found}} found${{f.lost ? '  (LOST)' : ''}}`; }}
s.oninput = () => show(+s.value);
document.getElementById('prev').onclick = () => show(+s.value - 1);
document.getElementById('next').onclick = () => show(+s.value + 1);
let timer = null;
document.getElementById('play').onclick = e => {{ if (timer) {{ clearInterval(timer); timer = null; e.target.textContent = 'play'; }}
  else {{ timer = setInterval(() => show(+s.value + 1 < frames.length ? +s.value + 1 : 0), 250); e.target.textContent = 'pause'; }} }};
document.onkeydown = e => {{ if (e.key === 'ArrowLeft') show(+s.value - 1); if (e.key === 'ArrowRight') show(+s.value + 1); }};
show(0);
</script></body></html>""", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", type=date.fromisoformat, required=True, help="first day (UTC)")
    p.add_argument("--end", type=date.fromisoformat, default=None, help="last day (default: --start)")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "landmarks")
    p.add_argument("--preview", action="store_true", help="draw the spots on every photo + index.html per day")
    p.add_argument("--table", action="store_true", help="with --preview/--video: also the alignment table (red, az login)")
    p.add_argument("--video", action="store_true", help="timelapse MP4 of the whole range (data/landmarks/)")
    p.add_argument("--fps", type=int, default=15, help="timelapse frames per second")
    p.add_argument("--verbose", action="store_true", help="print one line every 10 photos")
    args = p.parse_args()
    end = args.end or args.start

    load_dotenv(ROOT / "tools" / ".env")
    lm = json.loads((ROOT / "label_studio" / "landmarks.json").read_text(encoding="utf-8"))
    rois = json.loads((ROOT / "label_studio" / "rois.json").read_text(encoding="utf-8"))
    size = lm["image_size"]
    ref = to_px(lm["landmarks"], size)
    anchors = {b: {k: v for k, v in to_px(a["landmarks"], size).items() if k in ref} for b, a in lm["anchors"].items()}
    anchors[lm["source_image"]] = ref  # the reference photo is an anchor too
    container = BlobServiceClient.from_connection_string(os.environ["AZURE_STORAGE_CONNECTION_STRING"]) \
        .get_container_client(os.getenv("PW_CONTAINER", "raw"))
    download = lambda b: container.download_blob(b).readall()  # noqa: E731
    tracker = Tracker(ref, anchors, download)
    tracker.start(near=f"x/parkwatch_{args.start:%Y%m%d}T120000Z.jpg")
    print(f"Start positions: {tracker.nearest_anchor(f'x/parkwatch_{args.start:%Y%m%d}T120000Z.jpg')}")
    table = load_table() if (args.preview or args.video) and args.table else None
    centres = spot_centres(rois, size)

    streak, prev_blob = None, ""
    streaks, validation = [], []
    video, video_path = None, args.out / f"timelapse_{args.start}_{end}.mp4"
    day = args.start
    while day <= end:
        prefix = f"{day:%Y/%m/%d}/"
        blobs = sorted(b.name for b in container.list_blobs(name_starts_with=prefix) if b.name.endswith(".jpg"))
        out_dir = args.out / f"{day}"
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.preview:
            (out_dir / "preview").mkdir(exist_ok=True)
        rows, frames = [], []
        with ThreadPoolExecutor(8) as pool:  # downloads ahead, in order
            for i, (blob, data) in enumerate(zip(blobs, pool.map(download, blobs))):
                result = tracker.step(blob, data)
                lost = result["status"] == "lost"
                if lost:
                    streak = streak or {"first": blob, "n": 0}
                    streak["n"] += 1
                elif streak:
                    streaks.append({**streak, "last": prev_blob})
                    streak = None
                if result["status"] == "anchor":
                    before, clicked = result["before_anchor"], anchors[blob]
                    if before is None:
                        validation.append((blob, None, None))
                    else:
                        err = float(np.mean([np.hypot(*(before[k] - clicked[k])) for k in clicked]))
                        validation.append((blob, err, spot_error(ref, before, clicked, centres)))
                prev_blob = blob

                positions, found = tracker.positions, set(result["found"])
                hhmmss = blob.rsplit("T", 1)[1][:6]
                row = {"blob": blob, "status": result["status"], "n_found": len(found), "lost": int(lost),
                       "score": round(result["score"], 3)}
                for sid, w in spot_warps(ref, positions, centres).items():
                    sh = apply(w, centres[sid][None])[0] - centres[sid]
                    row[f"{sid}_dx"], row[f"{sid}_dy"] = round(float(sh[0]), 1), round(float(sh[1]), 1)
                rows.append(row)
                if args.preview or args.video:
                    name = f"{hhmmss}.jpg"
                    frame = draw(cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR), rois, ref, positions,
                                 found, size, table.warps(blob) if table and blob in table.rows else None,
                                 f"{day} {hhmmss[:2]}:{hhmmss[2:4]} UTC  {len(found)} landmarks found"
                                 + ("  LOST: positions of the last tracked photo" if lost else ""),
                                 out_dir / "preview" / name if args.preview else None)
                    if args.preview:
                        frames.append({"file": f"preview/{name}", "time": f"{hhmmss[:2]}:{hhmmss[2:4]}",
                                       "found": len(found), "lost": lost})
                    if args.video:
                        if video is None:
                            video = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
                                                    (frame.shape[1], frame.shape[0]))
                        video.write(frame)
                if args.verbose and (i % 10 == 0 or lost):
                    print(f"  {hhmmss[:2]}:{hhmmss[2:4]} {len(found):3} found {row['score']:.2f}  "
                          + ("LOST" if lost else "  ".join(f"{s}({row[f'{s}_dx']:.0f},{row[f'{s}_dy']:.0f})" for s in centres)))

        if rows:
            with open(out_dir / "tracking.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            if args.preview:
                write_viewer(out_dir / "index.html", f"{day}", frames, table is not None)
            noon = min(rows, key=lambda r: abs(int(r["blob"].rsplit("T", 1)[1][:4]) - 1100))
            n_lost = sum(r["lost"] for r in rows)
            print(f"{day}: {len(rows) - n_lost}/{len(rows)} tracked | ~11:00 UTC "
                  + "  ".join(f"{s}({noon[f'{s}_dx']:.0f},{noon[f'{s}_dy']:.0f})" for s in centres))
        else:
            print(f"{day}: no photos")
        day += timedelta(days=1)
    if streak:
        streaks.append({**streak, "last": prev_blob})
    if video is not None:
        video.release()
        print(f"Timelapse -> {video_path}")

    print(f"\nLost stretches ({len(streaks)}): photos without enough landmarks. Darkness at dawn/dusk is "
          "expected; in daylight an anchor on the first lost photo fixes it.")
    for s in streaks:
        print(f"  {s['first']} .. {s['last'].rsplit('_', 1)[1]}  ({s['n']} photos)")
    if validation:
        print("\nValidation against the anchors (tracking before resetting to the clicks):")
        for blob, err, spots in validation:
            if err is None:
                print(f"  {blob}: LOST before the anchor")
            else:
                print(f"  {blob}: landmarks {err:.1f} px | " + "  ".join(f"{s} {e:.0f}px" for s, e in spots.items()))
    if args.preview:
        print(f"\nViewers: {args.out}/<day>/index.html")


if __name__ == "__main__":
    main()
