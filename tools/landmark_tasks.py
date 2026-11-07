"""ParkWatch — creates tasks (photos) in the Label Studio landmarks project.

Each task shows a Blob photo (azure-blob://, resolved by the project's Azure source storage
with pre-signed URLs, no sync) to click the landmarks on it (label_studio/landmarks_config.xml).

Usage (repo root; LS_URL and LS_TOKEN in tools/.env):
  uv run python tools/landmark_tasks.py --project 7 --reference           # the rois.json photo
  uv run python tools/landmark_tasks.py --project 7 2026/11/06/parkwatch_20261106T065927Z.jpg ...
  uv run python tools/landmark_tasks.py --project 7 --at "2026-10-29 11:00" "2026-11-04 11:00"
      (--at: the photo nearest to that UTC time; needs AZURE_STORAGE_CONNECTION_STRING in tools/.env)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))
from ls_client import import_tasks, ls_headers  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("blobs", nargs="*", help="YYYY/MM/DD/parkwatch_<ts>Z.jpg")
    p.add_argument("--project", type=int, required=True, help="id of the parkwatch-landmarks project")
    p.add_argument("--reference", action="store_true", help="also the reference photo of rois.json")
    p.add_argument("--at", nargs="*", default=[], help='"YYYY-MM-DD HH:MM" (UTC): the photo nearest to that time')
    p.add_argument("--container", default="raw")
    args = p.parse_args()

    load_dotenv(ROOT / "tools" / ".env")
    blobs = list(args.blobs)
    if args.reference:
        blobs.insert(0, json.loads((ROOT / "label_studio" / "rois.json").read_text(encoding="utf-8"))["source_image"])
    if args.at:
        from azure.storage.blob import BlobServiceClient

        from alignment import blob_time

        service = BlobServiceClient.from_connection_string(os.environ["AZURE_STORAGE_CONNECTION_STRING"])
        container = service.get_container_client(args.container)
        for at in args.at:
            t = datetime.strptime(at, "%Y-%m-%d %H:%M")
            day = [b.name for b in container.list_blobs(name_starts_with=f"{t:%Y/%m/%d}/") if b.name.endswith(".jpg")]
            if not day:
                raise SystemExit(f"No photos on {t:%Y-%m-%d}")
            blobs.append(min(day, key=lambda b: abs(blob_time(b) - t)))
    if not blobs:
        raise SystemExit("No photos: pass blob paths and/or --reference")
    ls_url = os.environ["LS_URL"].rstrip("/")
    tasks = [{"data": {"image": f"azure-blob://{args.container}/{b}", "blob": b}} for b in blobs]
    import_tasks(ls_url, ls_headers(ls_url, os.environ["LS_TOKEN"]), args.project, tasks)
    for b in blobs:
        print(f"  {b}")


if __name__ == "__main__":
    main()
