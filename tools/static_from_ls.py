"""ParkWatch — converts the Label Studio `static` polygons into label_studio/static_mask.json.

Background regions that never change (facades): the camera drift is measured only inside
them, without sky, sun, glass reflections or cars. Same point format as rois.json (% of the
image) and same reference photo.

Usage:
  uv run python tools/static_from_ls.py export.json
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from rois_from_ls import pick_annotation

PHOTO_RE = re.compile(r"parkwatch_(\d{4})(\d{2})(\d{2})T\d{6}Z\.jpg$")


def blob_path(task: dict) -> str:
    """Blob path of the photo (YYYY/MM/DD/parkwatch_<ts>Z.jpg), as in rois.json.

    If the task was created by uploading the image to Label Studio, data.image is /data/upload/..../<hash>-parkwatch_<ts>Z.jpg."""
    data = task.get("data", {})
    if data.get("blob"):
        return data["blob"]
    m = PHOTO_RE.search(data.get("image", ""))
    if not m:
        raise SystemExit(f"Cannot recognise the task photo: {data.get('image')}")
    return f"{m.group(1)}/{m.group(2)}/{m.group(3)}/{m.group(0)}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("export", type=Path, help="Label Studio JSON export (parkwatch-rois project)")
    p.add_argument("-o", "--out", type=Path, default=Path("label_studio/static_mask.json"))
    args = p.parse_args()

    task, annotation = pick_annotation(json.loads(args.export.read_text(encoding="utf-8")))
    regions = [r for r in annotation["result"]
               if r.get("type") == "polygonlabels" and r["value"].get("polygonlabels", [""])[0] == "static"
               and len(r["value"]["points"]) >= 3]
    if not regions:
        raise SystemExit("The annotation has no `static` polygons.")
    mask = {
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source_image": blob_path(task),
        "image_size": [regions[0].get("original_width"), regions[0].get("original_height")],
        "regions": [[[round(x, 3), round(y, 3)] for x, y in r["value"]["points"]] for r in regions],
    }
    args.out.write_text(json.dumps(mask, indent=2), encoding="utf-8")
    print(f"{len(mask['regions'])} static regions -> {args.out} (reference {mask['source_image']})")


if __name__ == "__main__":
    main()
