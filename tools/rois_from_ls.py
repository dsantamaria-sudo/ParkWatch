"""ParkWatch — converts the export of the Label Studio ROI project into rois.json.

Flow (once, after fixing the camera in place):
  1. Label Studio project with label_studio/roi_config.xml + reference photo.
  2. Draw one polygon per spot and Submit.
  3. Export -> "JSON" format (not JSON-MIN) -> this script.

Usage:
  uv run python tools/rois_from_ls.py export.json -o label_studio/rois.json
  uv run python tools/rois_from_ls.py export.json --sort x   # ids ordered left->right

Adjusting existing ROIs: if the task was created with the current ROIs as prediction, the
regions carry ids p01..pNN and those ids are kept (labels already made remain valid).

Points are stored in % of the image width/height (what Label Studio uses), so they work at
any resolution as long as the framing does not change.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date, datetime, timezone
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw

OVERLAP_WARN_PCT = 25  # above this, the spot crop mixes in a lot of other spots
SPOT_ID_RE = re.compile(r"p\d{2,}")


def centroid(points: list[list[float]]) -> tuple[float, float]:
    xs, ys = zip(*points)
    return sum(xs) / len(xs), sum(ys) / len(ys)


def overlap_report(spots: list[dict], image_size: list, width: int = 1000) -> list[tuple[str, float, str]]:
    """For each spot: % of its area covered by other spots and which one overlaps it most.

    Rasterises the polygons (in %) on a grid with the aspect ratio of the image.
    """
    w, h = image_size if all(image_size) else (4, 3)
    size = (width, max(1, round(width * h / w)))

    def mask(points: list[list[float]]) -> Image.Image:
        m = Image.new("L", size, 0)
        ImageDraw.Draw(m).polygon([(x / 100 * size[0], y / 100 * size[1]) for x, y in points], fill=255)
        return m

    def area(m: Image.Image) -> int:
        return m.histogram()[255]

    masks = {s["id"]: mask(s["points"]) for s in spots}
    report = []
    for sid, m in masks.items():
        own = area(m) or 1
        others = Image.new("L", size, 0)
        worst_id, worst = "-", 0
        for oid, om in masks.items():
            if oid == sid:
                continue
            inter = area(ImageChops.logical_and(m.convert("1"), om.convert("1")).convert("L"))
            if inter > worst:
                worst_id, worst = oid, inter
            others = ImageChops.lighter(others, om)
        covered = area(ImageChops.logical_and(m.convert("1"), others.convert("1")).convert("L"))
        report.append((sid, 100 * covered / own, f"{worst_id} ({100 * worst / own:.0f}%)"))
    return report


def pick_annotation(export: list | dict) -> tuple[dict, dict]:
    """Returns (task, annotation): the newest non-cancelled annotation of the whole export."""
    tasks = export if isinstance(export, list) else [export]
    candidates = [
        (t, a) for t in tasks for a in t.get("annotations", []) if not a.get("was_cancelled")
    ]
    if not candidates:
        sys.exit("The export has no annotations. Export in JSON format (not JSON-MIN) after Submit.")
    task, annotation = max(candidates, key=lambda ta: ta[1].get("updated_at") or ta[1].get("created_at") or "")
    if len(candidates) > 1:
        print(f"Warning: {len(candidates)} annotations in the export, using the newest (task {task.get('id')}).",
              file=sys.stderr)
    return task, annotation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("export", type=Path, help="Label Studio JSON export")
    parser.add_argument("-o", "--out", type=Path, default=Path("label_studio/rois.json"))
    parser.add_argument(
        "--sort",
        choices=["drawn", "x", "y"],
        default="drawn",
        help="id order: as drawn, left->right (x) or top->bottom (y)",
    )
    parser.add_argument("--version", default=date.today().isoformat(), help="ROI version (default: today)")
    parser.add_argument(
        "--only",
        help="comma-separated ids (e.g. p04,p05): only those spots are taken from the export, "
        "the rest are kept as they are in the current --out",
    )
    args = parser.parse_args()

    task, annotation = pick_annotation(json.loads(args.export.read_text(encoding="utf-8")))
    # spots only: `static` polygons (background used to measure drift) go to static_from_ls.py
    regions = [r for r in annotation["result"]
               if r.get("type") == "polygonlabels" and r["value"].get("polygonlabels", ["spot"])[0] == "spot"]
    if not regions:
        sys.exit("The annotation has no polygons (PolygonLabels).")

    polygons = []
    for r in regions:
        points = r["value"]["points"]
        if len(points) < 3:
            print(f"Warning: polygon {r.get('id')} has fewer than 3 points, skipped.", file=sys.stderr)
            continue
        polygons.append((r.get("id", ""), points))

    # When adjusting existing ROIs (task created with the ROIs as prediction) the regions already
    # carry their ids p01..pNN: they are kept so that labelled spots are not reassigned.
    ids = [rid for rid, _ in polygons]
    keep_ids = args.sort == "drawn" and len(set(ids)) == len(ids) and all(SPOT_ID_RE.fullmatch(i) for i in ids)
    if keep_ids:
        polygons.sort(key=lambda p: p[0])
        print("Existing spot ids kept.")
    elif args.sort == "x":
        polygons.sort(key=lambda p: centroid(p[1]))
    elif args.sort == "y":
        polygons.sort(key=lambda p: centroid(p[1])[::-1])
    if not keep_ids:
        polygons = [(f"p{i:02d}", points) for i, (_, points) in enumerate(polygons, start=1)]

    if args.only:
        only = [s.strip() for s in args.only.split(",") if s.strip()]
        edited = dict(polygons)
        missing = [s for s in only if s not in edited]
        if missing:
            sys.exit(f"--only: {missing} are not in the annotation")
        base = json.loads(args.out.read_text(encoding="utf-8"))
        polygons = [(s["id"], edited[s["id"]] if s["id"] in only else s["points"]) for s in base["spots"]]
        print(f"Only {only} are updated; the rest is kept from {args.out}.")

    first = regions[0]
    rois = {
        "version": args.version,
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        # data.blob when available: data.image may be a signed (SAS) URL that must not end up in git
        "source_image": task.get("data", {}).get("blob") or task.get("data", {}).get("image"),
        "image_size": [first.get("original_width"), first.get("original_height")],
        "spots": [
            {"id": spot_id, "points": [[round(x, 3), round(y, 3)] for x, y in points]}
            for spot_id, points in polygons
        ],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rois, indent=2), encoding="utf-8")
    print(f"{len(rois['spots'])} spots -> {args.out} (version {rois['version']})")
    print(f"  {'spot':<6} {'centre':<16} {'overlap':>7}  largest overlap with")
    flagged = 0
    for spot, (_, covered, worst) in zip(rois["spots"], overlap_report(rois["spots"], rois["image_size"])):
        cx, cy = centroid(spot["points"])
        warn = "  <-- review" if covered > OVERLAP_WARN_PCT else ""
        flagged += bool(warn)
        print(f"  {spot['id']:<6} ({cx:5.1f}%, {cy:5.1f}%) {covered:6.0f}%  {worst}{warn}")
    if flagged:
        print(f"{flagged} spot(s) with more than {OVERLAP_WARN_PCT}% of their area overlapping others.")


if __name__ == "__main__":
    main()
