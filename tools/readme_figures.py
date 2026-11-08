"""ParkWatch — README figure (docs/images/alignment_before_after.png).

Zoom on spots p04-p05 of a photo taken after the camera moved: fixed ROIs on the raw photo |
ROIs moved with the landmarks (every spot with its own warp, from the latest alignment table
written by align_photos) | reference photo.

Usage (repo root; photos read with the connection string in tools/.env, the table with az login):
  uv run python tools/readme_figures.py
  uv run python tools/readme_figures.py --photo 2026/11/03/parkwatch_20261103T155920Z.jpg
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from azure.storage.blob import BlobServiceClient  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))
from alignment import Alignment, describe, transform_points  # noqa: E402

COLORS = {"raw": "#e74c3c", "aligned": "#2ecc71", "reference": "#f1c40f"}


def draw_rois(image: Image.Image, rois: dict, color: str, ids: list[str], width: int = 5,
              warps: dict | None = None) -> Image.Image:
    out = image.copy()
    d = ImageDraw.Draw(out)
    w, h = out.size
    for spot in rois["spots"]:
        if spot["id"] in ids:
            points = spot["points"] if not warps else transform_points(spot["points"], warps[spot["id"]], (w, h))
            pts = [(x / 100 * w, y / 100 * h) for x, y in points]
            d.line(pts + [pts[0]], fill=color, width=width, joint="curve")
    return out


def zoom_box(rois: dict, ids: list[str], size: tuple[int, int], margin: float = 0.5) -> tuple[int, int, int, int]:
    w, h = size
    pts = [(x / 100 * w, y / 100 * h) for s in rois["spots"] if s["id"] in ids for x, y in s["points"]]
    x0, x1 = min(p[0] for p in pts), max(p[0] for p in pts)
    y0, y1 = min(p[1] for p in pts), max(p[1] for p in pts)
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    return int(max(0, x0 - mx)), int(max(0, y0 - my)), int(min(w, x1 + mx)), int(min(h, y1 + my))


def panels(images: list[Image.Image], titles: list[str], colors: list[str], path: Path, suptitle: str) -> None:
    fig, axes = plt.subplots(1, len(images), figsize=(5.2 * len(images), 3.9))
    for ax, im, title, color in zip(axes, images, titles, colors):
        ax.imshow(im)
        ax.set_title(title, fontsize=12, color="#222", pad=8)
        ax.set_xticks([]), ax.set_yticks([])
        for side in ax.spines.values():
            side.set_edgecolor(color)
            side.set_linewidth(3)
    fig.suptitle(suptitle, fontsize=13, y=0.99)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rois", type=Path, default=Path("label_studio/rois.json"))
    p.add_argument("--photo", default="2026/11/03/parkwatch_20261103T155920Z.jpg", help="example photo (after the camera fell)")
    p.add_argument("--zoom", default="p04,p05")
    p.add_argument("--out", type=Path, default=Path("docs/images"))
    args = p.parse_args()

    load_dotenv(Path(__file__).with_name(".env"))
    container = BlobServiceClient.from_connection_string(
        os.environ["AZURE_STORAGE_CONNECTION_STRING"]
    ).get_container_client(os.getenv("PW_CONTAINER", "raw"))
    rois = json.loads(args.rois.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)

    ref_bytes = container.download_blob(rois["source_image"]).readall()
    photo_bytes = container.download_blob(args.photo).readall()
    reference = Image.open(io.BytesIO(ref_bytes)).convert("RGB")
    photo = Image.open(io.BytesIO(photo_bytes)).convert("RGB")
    import csv

    from azure_utils import latest_asset_file

    os.environ.setdefault("AZUREML_ARM_SUBSCRIPTION", "48ee8f92-9a0c-4ecd-b6ce-83845f85bf75")
    os.environ.setdefault("AZUREML_ARM_RESOURCEGROUP", "parkwatch")
    os.environ.setdefault("AZUREML_ARM_WORKSPACE_NAME", "parkwatch-workspace")
    content, _ = latest_asset_file("parkwatch-alignment", "alignment.csv")
    table = Alignment(list(csv.DictReader(io.StringIO(content.decode("utf-8")))))
    if args.photo not in table.rows:
        raise SystemExit(f"{args.photo} is not in the alignment table")
    warps = table.warps(args.photo)
    ids = args.zoom.split(",")
    drift = describe(warps[ids[0]])
    print(f"{args.photo} ({ids[0]}): dx {drift['dx']:+.1f} px, dy {drift['dy']:+.1f} px, rotation {drift['angle']:+.2f}°")

    box = zoom_box(rois, ids, reference.size)
    moved = {"spots": [{"id": sp["id"], "points": transform_points(sp["points"], warps[sp["id"]], photo.size)}
                       for sp in rois["spots"]]}
    moved_box = zoom_box(moved, ids, photo.size)  # the moved spots are elsewhere on the raw photo
    panels(
        [draw_rois(photo, rois, COLORS["raw"], ids).crop(box),
         draw_rois(photo, rois, COLORS["aligned"], ids, warps=warps).crop(moved_box),
         draw_rois(reference, rois, COLORS["reference"], ids).crop(box)],
        ["Fixed ROIs, no alignment", "ROIs moved with the landmarks", "Reference photo"],
        [COLORS["raw"], COLORS["aligned"], COLORS["reference"]],
        args.out / "alignment_before_after.png",
        f"Spots p04–p05 after the camera moved ({drift['dx']:+.0f}, {drift['dy']:+.0f}) px and {drift['angle']:+.1f}°",
    )
    print(f"figure -> {args.out / 'alignment_before_after.png'}")


if __name__ == "__main__":
    main()
