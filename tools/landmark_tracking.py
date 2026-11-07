"""ParkWatch — experiment: automatic tracking of the landmarks over a range of days.

Every landmark (label_studio/landmarks.json) is searched in each photo, in time order, near
where it was in the previous photo (the camera moves little in 3 minutes, and hardly at
night: each day starts where the previous one ended): normalised cross-correlation of a
patch around the landmark, cut from a TEMPLATE photo, the anchor nearest in time (or the
reference). A match is accepted if it is good and it agrees with the rigid movement
(rotation + shift) of the others since the previous photo. A landmark not found is moved with
that same movement, so every spot always keeps its landmarks around it; after MISSED_WIDE
photos without it, it is searched in a wider radius. With fewer than MIN_LANDMARKS matches the photo is LOST: the positions
of the last tracked photo are kept and it is reported (in the pipeline: ask for an anchor).

The reference photo of rois.json counts as an anchor. On an anchor photo the tracking is first compared with the clicks (validation) and then the
positions are reset to the clicks (as the pipeline will do).

Every spot gets its own shift from the landmarks around it (local similarity, fisheye lens).

Output in data/landmarks/<day>/ (outside git): tracking.csv (one row per photo: landmarks
found, mean match score, shift of every spot) and, with --preview, every photo drawn with the
moved spots (green; red: current alignment table with --table) plus index.html to browse
them (local only: the photos never leave the machine).

With --video it writes the same frames, for the whole range, into a timelapse MP4
(data/landmarks/timelapse_<start>_<end>.mp4).

Usage (repo root; AZURE_STORAGE_CONNECTION_STRING in tools/.env):
  uv run python tools/landmark_tracking.py --start 2026-10-16 --end 2026-11-06
  uv run python tools/landmark_tracking.py --start 2026-10-16 --end 2026-11-07 --video --table
  uv run python tools/landmark_tracking.py --start 2026-11-06 --preview --table
"""
from __future__ import annotations

import argparse
import bisect
import csv
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
sys.path.insert(0, str(ROOT / "tools"))
from align_photos import gray  # noqa: E402  (same CLAHE as the pipeline)
from alignment import blob_time  # noqa: E402
from landmarks_from_ls import apply, fit_rigid, local_warp  # noqa: E402

PATCH = 40          # half size of the template patch (81x81 px)
RADIUS = 40         # search radius around the previous position (px)
RADIUS_LOST = 120   # wider search after a lost photo
MIN_SCORE = 0.5     # normalised cross-correlation
MAX_DISAGREE = 8    # px: a match further than this from the rigid movement of the others is rejected
SPREAD_ROTATION = 400  # px: the rotation between photos is only estimated from matches this spread out
MISSED_WIDE = 3     # photos without a landmark before searching it in RADIUS_LOST
MIN_LANDMARKS = 3


def to_px(points: dict, size) -> dict[str, np.ndarray]:
    w, h = size
    return {k: np.array([x / 100 * w, y / 100 * h]) for k, (x, y) in points.items()}


def find(image: np.ndarray, patch: np.ndarray, around: np.ndarray, radius: int) -> tuple[np.ndarray, float] | None:
    h, w = image.shape
    x0, y0 = int(around[0]) - PATCH - radius, int(around[1]) - PATCH - radius
    x1, y1 = int(around[0]) + PATCH + radius + 1, int(around[1]) + PATCH + radius + 1
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return None
    res = cv2.matchTemplate(image[y0:y1, x0:x1], patch, cv2.TM_CCOEFF_NORMED)
    _, score, _, (mx, my) = cv2.minMaxLoc(res)
    return np.array([x0 + mx + PATCH, y0 + my + PATCH], float), float(score)


def movement(src: np.ndarray, dst: np.ndarray) -> tuple[tuple, np.ndarray]:
    """Rigid movement between two photos. With matches close together (e.g. only the cluster at
    the base of the building) the rotation is unreliable and swings the far landmarks: then only
    the shift (median) is used; in 3 minutes the camera hardly rotates."""
    spread = np.hypot(*(src[:, None] - src[None]).transpose(2, 0, 1)).max()
    if spread >= SPREAD_ROTATION:
        return fit_rigid(src, dst)
    tx, ty = np.median(dst - src, axis=0)
    warp = (1.0, 0.0, float(tx), 0.0, 1.0, float(ty))
    return warp, np.hypot(*(apply(warp, src) - dst).T)


def coherent(previous: dict, found: dict) -> tuple[dict, tuple | None]:
    """Matches consistent with one rigid movement since the previous photo: the worst one is
    dropped while it is further than MAX_DISAGREE. Returns them and the movement."""
    keys = list(found)
    while len(keys) >= MIN_LANDMARKS:
        warp, residuals = movement(np.array([previous[k] for k in keys]), np.array([found[k] for k in keys]))
        worst = int(np.argmax(residuals))
        if residuals[worst] <= MAX_DISAGREE:
            return {k: found[k] for k in keys}, warp
        keys.pop(worst)
    return {}, None


def spot_error(ref: dict, tracked: dict, clicked: dict, centres: dict) -> dict[str, float]:
    """Distance between the spot centres moved with the tracking and with the clicks."""
    src_t, dst_t = np.array([ref[k] for k in tracked]), np.array(list(tracked.values()))
    keys = [k for k in clicked if k in ref]
    src_c, dst_c = np.array([ref[k] for k in keys]), np.array([clicked[k] for k in keys])
    return {sid: float(np.hypot(*(apply(local_warp(src_t, dst_t, c[0]), c)[0]
                                  - apply(local_warp(src_c, dst_c, c[0]), c)[0])))
            for sid, c in centres.items()}


class Templates:
    """Patches of every landmark cut from the anchor nearest in time to the photo."""

    def __init__(self, container, ref: dict, sources: dict[str, dict]):
        self.container, self.ref, self.sources = container, ref, sources
        self.times = sorted((blob_time(b), b) for b in sources)
        self.cache: dict[str, dict] = {}

    def nearest(self, blob: str) -> str:
        t = blob_time(blob)
        i = bisect.bisect_left(self.times, (t, ""))
        return min((self.times[j] for j in (i - 1, i) if 0 <= j < len(self.times)), key=lambda x: abs(x[0] - t))[1]

    def patches(self, blob: str) -> dict[str, np.ndarray]:
        source = self.nearest(blob)
        if source not in self.cache:
            image = gray(self.container.download_blob(source).readall())
            out = {}
            for k, (x, y) in self.sources[source].items():
                patch = image[int(y) - PATCH:int(y) + PATCH + 1, int(x) - PATCH:int(x) + PATCH + 1]
                if k in self.ref and patch.shape == (2 * PATCH + 1, 2 * PATCH + 1):
                    out[k] = patch
            self.cache = {source: out}  # one template in memory at a time
            print(f"  template: {source} ({len(out)} landmarks)")
        return self.cache[source]


def load_table():
    import io

    from alignment import Alignment
    from azure_utils import latest_asset_file

    os.environ.setdefault("AZUREML_ARM_SUBSCRIPTION", "48ee8f92-9a0c-4ecd-b6ce-83845f85bf75")
    os.environ.setdefault("AZUREML_ARM_RESOURCEGROUP", "parkwatch")
    os.environ.setdefault("AZUREML_ARM_WORKSPACE_NAME", "parkwatch-workspace")
    content, version = latest_asset_file("parkwatch-alignment", "alignment.csv")
    print(f"Current table: parkwatch-alignment:{version}")
    return Alignment(list(csv.DictReader(io.StringIO(content.decode("utf-8")))))


def draw(image, rois, ref, positions, found, size, table_warp, title, path, width: int = 960):
    """Spots moved with the local model (green) and with the current table (red); landmarks found
    (yellow) or carried with the common movement (grey)."""
    w, h = size
    src = np.array([ref[k] for k in positions])
    dst = np.array(list(positions.values()))
    for spot in rois["spots"]:
        pts = np.array([[x / 100 * w, y / 100 * h] for x, y in spot["points"]])
        if table_warp is not None:
            cv2.polylines(image, [apply(table_warp, pts).astype(np.int32)], True, (60, 60, 230), 2)
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
    p.add_argument("--table", action="store_true", help="with --preview: also the current alignment table (red, az login)")
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
    templates = Templates(container, ref, anchors)
    table = load_table() if args.preview and args.table else None
    centres = {s["id"]: np.array([[sum(x for x, _ in s["points"]) / len(s["points"]) / 100 * size[0],
                                   sum(y for _, y in s["points"]) / len(s["points"]) / 100 * size[1]]])
               for s in rois["spots"]}

    # start: positions of the anchor (or reference) nearest in time to the first day
    first = f"{args.start:%Y/%m/%d}/parkwatch_{args.start:%Y%m%d}T120000Z.jpg"
    start_source = templates.nearest(first)
    start_pos = templates.sources[start_source]
    offset = np.median(np.array([start_pos[k] - ref[k] for k in start_pos]), axis=0)
    last = {k: start_pos.get(k, ref[k] + offset) for k in ref}  # landmarks not clicked there: common offset
    print(f"Start positions: {start_source}")

    lost, streak, prev_blob = False, None, ""
    video, video_path = None, args.out / f"timelapse_{args.start}_{end}.mp4"
    missed = {k: 0 for k in ref}
    streaks, validation = [], []
    download = lambda b: container.download_blob(b).readall()  # noqa: E731
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
                image = gray(data)
                patches = templates.patches(blob)
                found, scores = {}, []
                for k, patch in patches.items():
                    if k not in last:
                        continue
                    hit = find(image, patch, last[k], RADIUS_LOST if lost or missed[k] >= MISSED_WIDE else RADIUS)
                    if hit and hit[1] >= MIN_SCORE:
                        found[k] = hit[0]
                        scores.append(hit[1])
                found, motion = coherent(last, found)
                if len(found) >= MIN_LANDMARKS:
                    if lost and streak:
                        streaks.append({**streak, "last": prev_blob})
                        streak = None
                    lost = False
                    last = {k: found[k] if k in found else apply(motion, last[k][None])[0] for k in last}
                    missed = {k: 0 if k in found else missed[k] + 1 for k in missed}
                else:
                    if not lost:
                        streak = {"first": blob, "n": 0}
                    lost = True
                    streak["n"] += 1
                if blob in anchors:  # validation, then reset to the clicks
                    clicked = anchors[blob]
                    if lost:
                        validation.append((blob, None, None))
                    else:
                        errors = [np.hypot(*(last[k] - clicked[k])) for k in clicked if k in last]
                        validation.append((blob, float(np.mean(errors)), spot_error(ref, last, clicked, centres)))
                    common = [k for k in clicked if k in last]
                    motion, _ = fit_rigid(np.array([last[k] for k in common]), np.array([clicked[k] for k in common]))
                    last = {k: clicked[k] if k in clicked else apply(motion, last[k][None])[0] for k in last}
                    missed = {k: 0 for k in missed}
                    if lost and streak:
                        streaks.append({**streak, "last": blob})
                        streak = None
                    lost = False
                prev_blob = blob

                hhmmss = blob.rsplit("T", 1)[1][:6]
                row = {"blob": blob, "n_found": len(found), "lost": int(lost),
                       "score": round(float(np.mean(scores)) if scores else 0.0, 3)}
                src, dst = np.array([ref[k] for k in last]), np.array(list(last.values()))
                for sid, c in centres.items():
                    sh = apply(local_warp(src, dst, c[0]), c)[0] - c[0]
                    row[f"{sid}_dx"], row[f"{sid}_dy"] = round(float(sh[0]), 1), round(float(sh[1]), 1)
                rows.append(row)
                if args.preview or args.video:
                    name = f"{hhmmss}.jpg"
                    frame = draw(cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR), rois, ref, last, found,
                                 size, table.warp(blob) if table and blob in table.rows else None,
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
    if lost and streak:
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
