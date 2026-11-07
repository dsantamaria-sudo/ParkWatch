"""ParkWatch — landmarks and anchors from the Label Studio landmarks project -> label_studio/landmarks.json.

  - reference photo (the one of rois.json): position of every landmark (lm1..lm8)
  - any other photo: an ANCHOR, i.e. the exact camera position for that photo. A rigid
    transform reference -> photo (rotation + shift, no scale: the camera does not zoom) is
    fitted to the clicked landmarks; with 3 or more, the residual of each landmark shows a
    misplaced click.

The lens is a fisheye: a small camera rotation moves every part of the image differently, so
the shift is reported where it matters, at the centre of every spot, next to the distance to
the nearest landmark (far from any landmark the value is an extrapolation). With --compare the
same is computed with the current alignment table.

Also draws label_studio/landmarks.png (reference with the numbered landmarks and the spots) to
know which point is which when clicking anchors.

Usage (repo root; LS_URL, LS_TOKEN and AZURE_STORAGE_CONNECTION_STRING in tools/.env):
  uv run python tools/landmarks_from_ls.py --project 7
  uv run python tools/landmarks_from_ls.py --project 7 --compare   # vs the current alignment table (az login)
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml"))
sys.path.insert(0, str(ROOT / "tools"))
from alignment import describe  # noqa: E402
from ls_client import export_tasks  # noqa: E402
from static_from_ls import blob_path  # noqa: E402

RESIDUAL_WARN_PX = 4.0  # a landmark further than this from the fitted transform: check the click
OUTLIER_PX = 60.0       # a click this far from what the other landmarks predict is dropped (misclick)
SIGMA_PX = 200.0        # local model: landmark weight exp(-d^2 / 2 sigma^2), d = distance (best of 150-500 on 6/11)


def keypoints(task: dict) -> tuple[dict[str, list[float]], tuple[int, int]] | None:
    """{label: [x%, y%]} of the newest non-cancelled annotation, and the image size."""
    annotations = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
    if not annotations:
        return None
    annotation = max(annotations, key=lambda a: a.get("updated_at") or a.get("created_at") or "")
    points, size = {}, None
    for r in annotation["result"]:
        if r.get("type") != "keypointlabels":
            continue
        label = r["value"]["keypointlabels"][0]
        if label in points:
            print(f"  warning: {label} clicked twice in task {task['id']}, using the last one", file=sys.stderr)
        points[label] = [round(r["value"]["x"], 4), round(r["value"]["y"], 4)]
        size = (r["original_width"], r["original_height"])
    return (points, size) if points else None


def apply(warp, pts: np.ndarray) -> np.ndarray:
    m00, m01, m02, m10, m11, m12 = warp
    return np.stack([m00 * pts[:, 0] + m01 * pts[:, 1] + m02, m10 * pts[:, 0] + m11 * pts[:, 1] + m12], axis=1)


def fit_rigid(src: np.ndarray, dst: np.ndarray) -> tuple[tuple[float, ...], np.ndarray]:
    """Least-squares rotation + shift (px) mapping src onto dst. Returns the 2x3 warp
    (m00 m01 m02 m10 m11 m12, same convention as alignment.csv) and the residual of each point."""
    ms, md = src.mean(axis=0), dst.mean(axis=0)
    s, d = src - ms, dst - md
    theta = math.atan2((s[:, 0] * d[:, 1] - s[:, 1] * d[:, 0]).sum(), (s * d).sum())
    c, sn = math.cos(theta), math.sin(theta)
    tx, ty = md - np.array([c * ms[0] - sn * ms[1], sn * ms[0] + c * ms[1]])
    warp = (c, -sn, float(tx), sn, c, float(ty))
    return warp, np.hypot(*(apply(warp, src) - dst).T)


def local_warp(src: np.ndarray, dst: np.ndarray, at: np.ndarray) -> tuple[float, ...]:
    """Similarity (rotation + shift + scale) fitted around `at`: every landmark weighs
    exp(-d^2 / 2 sigma^2), d = its distance to `at`. The fisheye lens (and a camera that also
    shifts) moves every part of the image differently: the nearest landmarks tell best."""
    w = np.sqrt(np.exp(-np.hypot(*(src - at).T) ** 2 / (2 * SIGMA_PX ** 2)) + 1e-6)
    rows, rhs, weights = [], [], []
    for (x, y), (u, v), wi in zip(src, dst, w):
        rows += [[x, -y, 1, 0], [y, x, 0, 1]]
        rhs += [u, v]
        weights += [wi, wi]
    weights = np.array(weights)
    a, b, tx, ty = np.linalg.lstsq(np.array(rows) * weights[:, None], np.array(rhs) * weights, rcond=None)[0]
    return (float(a), float(-b), float(tx), float(b), float(a), float(ty))


def leave_one_out(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Error (px) predicting every landmark from the others: global rigid vs local model."""
    rigid, local = [], []
    for i in range(len(src)):
        keep = np.arange(len(src)) != i
        warp, _ = fit_rigid(src[keep], dst[keep])
        rigid.append(np.hypot(*(apply(warp, src[i:i + 1])[0] - dst[i])))
        local.append(np.hypot(*(apply(local_warp(src[keep], dst[keep], src[i]), src[i:i + 1])[0] - dst[i])))
    return np.array(rigid), np.array(local)


def spot_centres(rois: dict, w: int, h: int) -> dict[str, np.ndarray]:
    return {s["id"]: np.array([[sum(x for x, _ in s["points"]) / len(s["points"]) / 100 * w,
                                sum(y for _, y in s["points"]) / len(s["points"]) / 100 * h]])
            for s in rois["spots"]}


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
        kp = keypoints(task)
        if kp:
            marked[blob_path(task)] = kp
    if reference not in marked:
        raise SystemExit(f"The reference photo {reference} has no landmarks yet (task created with --reference?)")
    ref_points, size = marked.pop(reference)
    print(f"Reference {reference}: {len(ref_points)} landmarks ({', '.join(sorted(ref_points))})")
    if len(ref_points) < 3:
        print("  warning: at least 3 landmarks are needed (4-6 recommended, spread over the image)")

    w, h = size
    px = lambda pts, keys: np.array([[pts[k][0] / 100 * w, pts[k][1] / 100 * h] for k in keys])  # noqa: E731
    table = None
    if args.compare:
        os.environ.setdefault("AZUREML_ARM_SUBSCRIPTION", "48ee8f92-9a0c-4ecd-b6ce-83845f85bf75")
        os.environ.setdefault("AZUREML_ARM_RESOURCEGROUP", "parkwatch")
        os.environ.setdefault("AZUREML_ARM_WORKSPACE_NAME", "parkwatch-workspace")
        import csv

        from alignment import Alignment
        from azure_utils import latest_asset_file

        content, version = latest_asset_file("parkwatch-alignment", "alignment.csv")
        table = Alignment(list(csv.DictReader(io.StringIO(content.decode("utf-8")))))
        print(f"Compared with parkwatch-alignment:{version}")

    centres = spot_centres(rois, w, h)
    anchors = {}
    for blob, (points, _) in sorted(marked.items()):
        common = sorted(set(points) & set(ref_points), key=lambda k: int(k[2:]))
        if len(common) < 2:
            print(f"{blob}: only {len(common)} landmark(s) in common with the reference, skipped")
            continue
        src, dst = px(ref_points, common), px(points, common)
        while len(common) >= 4:  # misclicks: drop the worst while it is an outlier
            _, local_err = leave_one_out(src, dst)
            worst = int(local_err.argmax())
            if local_err[worst] <= OUTLIER_PX:
                break
            print(f"  warning: {blob}: {common[worst]} is {local_err[worst]:.0f} px from where the other "
                  "landmarks put it -> dropped (fix or delete that click)")
            points = {k: v for k, v in points.items() if k != common[worst]}
            common.pop(worst)
            src, dst = np.delete(src, worst, axis=0), np.delete(dst, worst, axis=0)
        warp, residuals = fit_rigid(src, dst)
        anchors[blob] = {"landmarks": points, "warp": [round(v, 6) for v in warp]}
        shift = (dst - src).mean(axis=0)
        print(f"{blob}: {len(common)} landmarks, mean shift ({shift[0]:.0f}, {shift[1]:.0f}) px, "
              f"rotation {describe(warp)['angle']:.2f}°")
        if len(common) >= 3:
            worst = max(zip(residuals, common))
            flag = "  <-- check that click" if worst[0] > RESIDUAL_WARN_PX else ""
            print(f"    residual: mean {residuals.mean():.1f} px, worst {worst[1]} {worst[0]:.1f} px{flag}")
        print("    shift of every landmark: " + "  ".join(
            f"{k} ({x:.0f},{y:.0f})" for k, (x, y) in zip(common, dst - src)))
        if len(common) >= 4:
            rigid_err, local_err = leave_one_out(src, dst)
            print(f"    leave-one-out error: global rigid mean {rigid_err.mean():.0f} px (worst {rigid_err.max():.0f}) | "
                  f"local mean {local_err.mean():.0f} px (worst {local_err.max():.0f}, "
                  f"{common[int(local_err.argmax())]})")
        in_table = table is not None and blob in table.rows
        print(f"    {'spot':5} {'nearest lm':>11} {'local':>14} {'global rigid':>14}"
              + (f" {'table':>14} {'table err':>9}" if in_table else ""))
        for sid, c in centres.items():
            local = apply(local_warp(src, dst, c[0]), c)[0] - c[0]
            rigid = apply(warp, c)[0] - c[0]
            near = np.hypot(*(src - c).T).min()
            line = f"    {sid:5} {near:8.0f} px ({local[0]:5.0f}, {local[1]:5.0f}) ({rigid[0]:5.0f}, {rigid[1]:5.0f})"
            if in_table:
                theirs = apply(table.warp(blob), c)[0] - c[0]
                line += f" ({theirs[0]:5.0f}, {theirs[1]:5.0f}) {np.hypot(*(theirs - local)):6.0f} px"
            print(line)

    out = {
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source_image": reference,
        "image_size": [w, h],
        "landmarks": dict(sorted(ref_points.items())),
        "anchors": anchors,
    }
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"{len(ref_points)} landmarks + {len(anchors)} anchors -> {args.out}")
    if not args.no_png:
        draw_reference(reference, ref_points, rois, args.out.with_suffix(".png"))


if __name__ == "__main__":
    main()
