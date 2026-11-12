"""ParkWatch — README figures (docs/images/).

model_input.png: a photo with its five spots (named) and the model input of each one (production
crop: polygon bbox + margin, letterboxed to the input size).

landmark_alignment.png: the camera alignment in three steps: reference photo with the spots and
the landmarks | a later photo with every landmark found again (arrows from the reference
position) and the spots where they would be without correction | the same photo with every spot
moved with its own warp. Positions and warps come from the latest alignment table written by
align_photos.

Usage (repo root; photos read with the connection string in tools/.env, the table with az login):
  uv run python tools/readme_figures.py
  uv run python tools/readme_figures.py --photo 2026/11/08/parkwatch_20261108T120233Z.jpg
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from azure.storage.blob import BlobServiceClient  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from alignment import IDENTITY, Alignment, align_image, transform_points  # noqa: E402
from crop_preview import network_input  # noqa: E402
from crops import spot_polygons  # noqa: E402

REFERENCE, UNCORRECTED, CURRENT = "#ffd60a", "#ff453a", "#30d158"
OUTLINE = [pe.withStroke(linewidth=3, foreground="black")]


def closed(points: np.ndarray) -> np.ndarray:
    return np.vstack([points, points[:1]])


def view(points: list[np.ndarray], size: tuple[int, int], pad: int) -> tuple[float, float, float, float]:
    """Box around the points with a padding, inside the image."""
    allp = np.vstack(points)
    (x0, y0), (x1, y1) = np.maximum(allp.min(0) - pad, 0), np.minimum(allp.max(0) + pad, size)
    return x0, x1, y1, y0


def model_input(photo: Image.Image, rois: dict, warps: dict, margin: float, size: int, path: Path) -> None:
    ids = [s["id"] for s in rois["spots"]]
    w, h = photo.size
    moved = {s["id"]: np.array([[x / 100 * w, y / 100 * h] for x, y in
                                transform_points(s["points"], warps.get(s["id"], IDENTITY), (w, h))])
             for s in rois["spots"]}
    polygons = spot_polygons(rois)
    fig = plt.figure(figsize=(15.5, 4.3))
    grid = fig.add_gridspec(1, 1 + len(ids), width_ratios=[3.1] + [1] * len(ids), wspace=0.06)
    ax = fig.add_subplot(grid[0])
    ax.imshow(photo)
    for sid, pts in moved.items():
        ax.plot(*closed(pts).T, color=CURRENT, lw=2.4)
        ax.text(*pts.mean(0), sid, color="white", fontsize=11, fontweight="bold", ha="center", va="center",
                path_effects=OUTLINE)
    x0, x1, y1, y0 = view(list(moved.values()), (w, h), 110)
    ax.set_xlim(x0, x1), ax.set_ylim(y1, y0)
    ax.set_title("One photo, five parking spots", fontsize=12)
    ax.set_xticks([]), ax.set_yticks([])
    for k, s in enumerate(rois["spots"], 1):
        ax = fig.add_subplot(grid[k])
        aligned = align_image(photo, warps.get(s["id"], IDENTITY))
        ax.imshow(network_input(aligned, s["points"], polygons[s["id"]], "keep", margin, size), interpolation="nearest")
        ax.set_title(s["id"], fontsize=12)
        ax.set_xticks([]), ax.set_yticks([])
    fig.suptitle(f"What the model sees: one {size}×{size} crop per spot", fontsize=13, y=0.98)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def landmark_alignment(reference: Image.Image, photo: Image.Image, rois: dict, table_rows: list[dict],
                       ref_blob: str, blob: str, warps: dict, path: Path) -> None:
    by = {r["blob"]: r for r in table_rows}
    names = sorted({k[:-2] for k in table_rows[0] if k.startswith("lm") and k.endswith("_x")}, key=lambda k: int(k[2:]))
    positions = lambda r: np.array([[float(r[f"{k}_x"]), float(r[f"{k}_y"])] for k in names])  # noqa: E731
    p_ref, p_now = positions(by[ref_blob]), positions(by[blob])
    w, h = reference.size
    to_px = lambda pts: np.array([[x / 100 * w, y / 100 * h] for x, y in pts])  # noqa: E731
    fixed = [to_px(s["points"]) for s in rois["spots"]]
    moved = [to_px(transform_points(s["points"], warps[s["id"]], (w, h))) for s in rois["spots"]]
    x0, x1, y1, y0 = view([p_ref, p_now, *fixed, *moved], (w, h), 70)
    shift = float(np.hypot(*(p_now - p_ref).T).mean())

    fig, axes = plt.subplots(1, 3, figsize=(17, 3.9))
    titles = [f"1 · Reference photo: spots drawn and {len(names)} landmarks clicked once",
              f"2 · A later photo: the camera moved ~{shift:.0f} px, every landmark is found again",
              "3 · Each spot follows the landmarks around it"]
    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=11.5, pad=8)
        ax.set_xlim(x0, x1), ax.set_ylim(y1, y0)
        ax.set_xticks([]), ax.set_yticks([])
    axes[0].imshow(reference)
    for pts in fixed:
        axes[0].plot(*closed(pts).T, color=REFERENCE, lw=2.2)
    axes[0].scatter(*p_ref.T, s=70, facecolors="none", edgecolors=REFERENCE, linewidths=2.2)
    axes[1].imshow(photo)
    for pts in fixed:
        axes[1].plot(*closed(pts).T, color=UNCORRECTED, lw=2, ls="--")
    axes[1].scatter(*p_ref.T, s=70, facecolors="none", edgecolors=REFERENCE, linewidths=1.8)
    for a, b in zip(p_ref, p_now):
        axes[1].annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", color=CURRENT, lw=2.2, shrinkA=0, shrinkB=0))
    axes[1].scatter(*p_now.T, s=28, color=CURRENT, zorder=3)
    axes[2].imshow(photo)
    for pts in fixed:
        axes[2].plot(*closed(pts).T, color=UNCORRECTED, lw=1.4, ls="--", alpha=0.75)
    for pts in moved:
        axes[2].plot(*closed(pts).T, color=CURRENT, lw=2.6)
    axes[2].scatter(*p_now.T, s=28, color=CURRENT, zorder=3)
    fig.text(0.5, 0.01, "yellow: reference   ·   red dashed: spots without correction   ·   "
                        "green: landmarks and spots in this photo", ha="center", fontsize=10.5, color="#333")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rois", type=Path, default=Path("label_studio/rois.json"))
    p.add_argument("--photo", default="2026/11/08/parkwatch_20261108T120233Z.jpg", help="example photo (camera moved)")
    p.add_argument("--crop-margin", type=float, default=0.10, help="production crop margin")
    p.add_argument("--input-size", type=int, default=96)
    p.add_argument("--out", type=Path, default=Path("docs/images"))
    args = p.parse_args()

    load_dotenv(Path(__file__).with_name(".env"))
    container = BlobServiceClient.from_connection_string(
        os.environ["AZURE_STORAGE_CONNECTION_STRING"]
    ).get_container_client(os.getenv("PW_CONTAINER", "raw"))
    rois = json.loads(args.rois.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    image = lambda blob: Image.open(io.BytesIO(container.download_blob(blob).readall())).convert("RGB")  # noqa: E731

    from azure_utils import latest_asset_file

    os.environ.setdefault("AZUREML_ARM_SUBSCRIPTION", "48ee8f92-9a0c-4ecd-b6ce-83845f85bf75")
    os.environ.setdefault("AZUREML_ARM_RESOURCEGROUP", "parkwatch")
    os.environ.setdefault("AZUREML_ARM_WORKSPACE_NAME", "parkwatch-workspace")
    content, version = latest_asset_file("parkwatch-alignment", "alignment.csv")
    rows = list(csv.DictReader(io.StringIO(content.decode("utf-8"))))
    table = Alignment(rows)
    for blob in (args.photo, rois["source_image"]):
        if blob not in table.rows:
            raise SystemExit(f"{blob} is not in the alignment table")
    warps = table.warps(args.photo)
    reference, photo = image(rois["source_image"]), image(args.photo)
    print(f"{args.photo} | alignment table v{version}")

    model_input(photo, rois, warps, args.crop_margin, args.input_size, args.out / "model_input.png")
    print(f"figure -> {args.out / 'model_input.png'}")
    landmark_alignment(reference, photo, rois, rows, rois["source_image"], args.photo, warps,
                       args.out / "landmark_alignment.png")
    print(f"figure -> {args.out / 'landmark_alignment.png'}")


if __name__ == "__main__":
    main()
