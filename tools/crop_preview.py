"""ParkWatch — what ParkNet sees: the input of every spot of one photo, for several crop margins.

Same code as production (ml/inference.py): the photo is straightened per spot with the latest
alignment table, cropped with the margin and letterboxed to the input size.
Pixels are shown enlarged (nearest), so the image is exactly the network input.

Usage (repo root; photo read with the connection string in tools/.env, the table with az login):
  uv run python tools/crop_preview.py --photo 2026/11/07/parkwatch_20261107T120000Z.jpg
  uv run python tools/crop_preview.py --at "2026-11-07 12:00"          # nearest photo to that UTC time
  uv run python tools/crop_preview.py --at "2026-11-07 12:00" --margins 0.10 0 0.25
-> data/previews/crops_<photo>.png
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from azure.storage.blob import BlobServiceClient  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))
from alignment import IDENTITY, Alignment, align_image, blob_time  # noqa: E402
from crops import crop_spot, letterbox  # noqa: E402


def nearest_photo(container, at: datetime) -> str:
    blobs = [b.name for b in container.list_blobs(name_starts_with=f"{at:%Y/%m/%d}/") if b.name.endswith(".jpg")]
    if not blobs:
        raise SystemExit(f"no photos on {at:%Y-%m-%d}")
    return min(blobs, key=lambda b: abs((blob_time(b) - at).total_seconds()))


def network_input(aligned: Image.Image, points: list, margin: float, size: int) -> Image.Image:
    return letterbox(crop_spot(aligned, points, margin), size)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--photo", help="blob, e.g. 2026/11/07/parkwatch_20261107T120000Z.jpg")
    src.add_argument("--at", help='"YYYY-MM-DD HH:MM" UTC: nearest photo')
    p.add_argument("--margins", nargs="+", type=float, default=[0.10, 0.0, 0.25], help="crop margins to compare")
    p.add_argument("--input-size", type=int, default=96)
    p.add_argument("--rois", type=Path, default=Path("label_studio/rois.json"))
    p.add_argument("--out", type=Path, default=Path("data/previews"))
    args = p.parse_args()

    load_dotenv(Path(__file__).with_name(".env"))
    container = BlobServiceClient.from_connection_string(
        os.environ["AZURE_STORAGE_CONNECTION_STRING"]).get_container_client(os.getenv("PW_CONTAINER", "raw"))
    blob = args.photo or nearest_photo(container, datetime.strptime(args.at, "%Y-%m-%d %H:%M"))
    photo = Image.open(io.BytesIO(container.download_blob(blob).readall())).convert("RGB")
    rois = json.loads(args.rois.read_text(encoding="utf-8"))

    from azure_utils import latest_asset_file

    os.environ.setdefault("AZUREML_ARM_SUBSCRIPTION", "48ee8f92-9a0c-4ecd-b6ce-83845f85bf75")
    os.environ.setdefault("AZUREML_ARM_RESOURCEGROUP", "parkwatch")
    os.environ.setdefault("AZUREML_ARM_WORKSPACE_NAME", "parkwatch-workspace")
    content, version = latest_asset_file("parkwatch-alignment", "alignment.csv")
    warps = Alignment(list(csv.DictReader(io.StringIO(content.decode("utf-8"))))).warps(blob)
    print(f"{blob} | alignment table v{version} | {'aligned' if warps else 'NOT in the table: no alignment'}")

    spots = rois["spots"]
    fig, axes = plt.subplots(len(spots), len(args.margins), figsize=(2.3 * len(args.margins), 2.4 * len(spots)), squeeze=False)
    for i, s in enumerate(spots):
        aligned = align_image(photo, warps.get(s["id"], IDENTITY))
        for j, margin in enumerate(args.margins):
            ax = axes[i][j]
            ax.imshow(network_input(aligned, s["points"], margin, args.input_size), interpolation="nearest")
            ax.set_xticks([]), ax.set_yticks([])
            if i == 0:
                ax.set_title(f"margin {margin:g}", fontsize=10)
            if j == 0:
                ax.set_ylabel(s["id"], fontsize=11)
    fig.suptitle(f"{blob} — network input {args.input_size}x{args.input_size}", fontsize=10)
    fig.tight_layout()
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / f"crops_{Path(blob).stem}.png"
    fig.savefig(out, dpi=110)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
