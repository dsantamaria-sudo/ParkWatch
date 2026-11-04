"""ParkWatch — visual examples of the alignment (camera drift) to review it by eye.

For each day it takes the best photo of the --local-time window (as camera_drift.py does),
measures it with the same code as the pipeline (ml/align_photos.measure) and straightens it
(ml/alignment.align_image). Saves in --out (default data/drift/preview/, outside git):

  <day>.png   top row: original with the fixed ROIs | straightened with the same ROIs |
              reference/straightened checkerboard (when well aligned, the lines of the
              buildings and the ground continue without steps from one tile to the next)
              bottom row: zoom on p04-p05, original | straightened | reference

Usage (repo root; reads the Storage with the connection string in tools/.env):
  uv run python tools/alignment_preview.py
  uv run python tools/alignment_preview.py --days 2026-10-29 2026-11-02
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import numpy as np
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))
from align_photos import gray, measure, static_mask  # noqa: E402
from alignment import align_image, describe  # noqa: E402
from camera_drift import pick  # noqa: E402  (same time window)

TILE = 100       # checkerboard tile size in px
PANEL_W = 800    # width of each panel in the final image


def draw_rois(image: Image.Image, rois: dict, color: str) -> Image.Image:
    out = image.copy()
    d = ImageDraw.Draw(out)
    w, h = out.size
    for spot in rois["spots"]:
        pts = [(x / 100 * w, y / 100 * h) for x, y in spot["points"]]
        d.line(pts + [pts[0]], fill=color, width=4)
        cx, cy = sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)
        d.text((cx - 12, cy - 8), spot["id"], fill=color)
    return out


def checkerboard(a: Image.Image, b: Image.Image) -> Image.Image:
    xa, xb = np.asarray(a), np.asarray(b)
    yy, xx = np.indices(xa.shape[:2])
    take_b = ((yy // TILE + xx // TILE) % 2).astype(bool)
    return Image.fromarray(np.where(take_b[..., None], xb, xa))


def zoom_box(rois: dict, ids: list[str], size: tuple[int, int], margin: float = 0.6) -> tuple[int, int, int, int]:
    w, h = size
    pts = [(x / 100 * w, y / 100 * h) for s in rois["spots"] if s["id"] in ids for x, y in s["points"]]
    x0, x1 = min(p[0] for p in pts), max(p[0] for p in pts)
    y0, y1 = min(p[1] for p in pts), max(p[1] for p in pts)
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    return int(max(0, x0 - mx)), int(max(0, y0 - my)), int(min(w, x1 + mx)), int(min(h, y1 + my))


def sheet(top: list[Image.Image], bottom: list[Image.Image], title: str) -> Image.Image:
    def fit(im: Image.Image) -> Image.Image:
        return im.resize((PANEL_W, round(im.height * PANEL_W / im.width)), Image.BILINEAR)

    top, bottom = [fit(i) for i in top], [fit(i) for i in bottom]
    h = 30 + max(i.height for i in top) + max(i.height for i in bottom)
    out = Image.new("RGB", (PANEL_W * len(top), h), "white")
    ImageDraw.Draw(out).text((10, 8), title, fill="black")
    for k, im in enumerate(top):
        out.paste(im, (k * PANEL_W, 30))
    y = 30 + max(i.height for i in top)
    for k, im in enumerate(bottom):
        out.paste(im, (k * PANEL_W, y))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rois", type=Path, default=Path("label_studio/rois.json"))
    p.add_argument("--mask", type=Path, default=Path("label_studio/static_mask.json"))
    p.add_argument("--days", nargs="*", default=[], help="YYYY-MM-DD (default: every day since 16/10)")
    p.add_argument("--local-time", type=time.fromisoformat, default=time(10, 0))
    p.add_argument("--candidates", type=int, default=5)
    p.add_argument("--zoom", default="p04,p05", help="spots shown in the zoom")
    p.add_argument("--out", type=Path, default=Path("data/drift/preview"))
    args = p.parse_args()

    load_dotenv(Path(__file__).with_name(".env"))
    container = BlobServiceClient.from_connection_string(
        os.environ["AZURE_STORAGE_CONNECTION_STRING"]
    ).get_container_client(os.getenv("PW_CONTAINER", "raw"))
    rois = json.loads(args.rois.read_text(encoding="utf-8"))
    ref_bytes = container.download_blob(rois["source_image"]).readall()
    ref_gray = gray(ref_bytes)
    reference = Image.open(io.BytesIO(ref_bytes)).convert("RGB")
    mask = static_mask(args.mask, ref_gray.shape)
    box = zoom_box(rois, args.zoom.split(","), reference.size)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.days:
        days = [date.fromisoformat(d) for d in args.days]
    else:
        days, d = [], date(2026, 10, 16)
        while d < datetime.now(timezone.utc).date():
            days.append(d)
            d += timedelta(days=1)

    for day in days:
        blobs = sorted(b.name for b in container.list_blobs(name_starts_with=f"{day:%Y/%m/%d}/") if b.name.endswith(".jpg"))
        best = None
        for blob in pick(blobs, day, args.local_time, 60, args.candidates):
            data = container.download_blob(blob).readall()
            warp, inliers = measure(ref_gray, gray(data), mask)
            if warp is not None and (best is None or inliers > best[0]):
                best = (inliers, blob, data, warp)
        if best is None:
            print(f"{day}: no measurement")
            continue
        inliers, blob, data, warp = best
        photo = Image.open(io.BytesIO(data)).convert("RGB")
        aligned = align_image(photo, tuple(float(v) for v in warp.flatten()))
        drift = describe(tuple(float(v) for v in warp.flatten()))
        title = (f"{blob}   dx {drift['dx']:+.1f} px  dy {drift['dy']:+.1f} px  rotation {drift['angle']:+.2f}  "
                 f"inliers {inliers}   |   original + fixed ROIs  |  straightened + ROIs  |  reference/straightened checkerboard")
        out = sheet(
            [draw_rois(photo, rois, "red"), draw_rois(aligned, rois, "lime"), checkerboard(reference, aligned)],
            [draw_rois(photo, rois, "red").crop(box), draw_rois(aligned, rois, "lime").crop(box),
             draw_rois(reference, rois, "yellow").crop(box)],
            title,
        )
        out.save(args.out / f"{day}.png")
        print(f"{day}: {blob}  dx {drift['dx']:+.1f}  dy {drift['dy']:+.1f}  rotation {drift['angle']:+.2f}  "
              f"inliers {inliers} -> {args.out / f'{day}.png'}")


if __name__ == "__main__":
    main()
