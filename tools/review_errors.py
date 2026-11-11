"""ParkWatch — review the errors of one spot from a training run (eval/cv_oof_predictions.csv).

Download the CSV from the run in MLflow (Artifacts -> eval/cv_oof_predictions.csv) and:
  uv run python tools/review_errors.py --csv cv_oof_predictions.csv --spot p05
  uv run python tools/review_errors.py --csv cv_oof_predictions.csv --spot p05 --kind false-free --sort time

Prints where the errors concentrate (day, UTC hour, image quality, error type) and writes contact
sheets to data/review/: each error as the network input (--outside / --margin of the run) next to
the photo around the spot with the aligned polygon, so a wrong label can be told from a model
mistake. Sorted by confidence (default: the most confident errors first, often wrong labels).

Photos are read with the connection string in tools/.env, the alignment table with az login.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from azure.storage.blob import BlobServiceClient  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from alignment import IDENTITY, Alignment, align_image, transform_points  # noqa: E402
from crop_preview import network_input  # noqa: E402
from crops import spot_polygons  # noqa: E402

KINDS = {"all": None, "false-free": ("occupied", "free"), "false-occupied": ("free", "occupied")}


def table(rows: list[dict], key, title: str, total: Counter) -> None:
    counts = Counter(key(r) for r in rows)
    print(f"\n{title}")
    for k in sorted(counts):
        n = total[k]
        print(f"  {k:>14}: {counts[k]:4d} errors / {n:5d} crops  ({counts[k] / n:6.1%})  {'#' * round(60 * counts[k] / max(counts.values()))}")


def context_view(photo: Image.Image, points: list, warp, pad: float = 0.6) -> Image.Image:
    """The photo around the spot with its polygon moved to this photo (the drift correction)."""
    w, h = photo.size
    pts = [(x / 100 * w, y / 100 * h) for x, y in transform_points(points, warp, (w, h))]
    out = photo.copy()
    ImageDraw.Draw(out).line(pts + [pts[0]], fill=(255, 40, 40), width=3)
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    mx, my = (max(xs) - min(xs)) * pad, (max(ys) - min(ys)) * pad
    return out.crop((int(max(0, min(xs) - mx)), int(max(0, min(ys) - my)),
                     int(min(w, max(xs) + mx)), int(min(h, max(ys) + my))))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, required=True, help="eval/cv_oof_predictions.csv of the run")
    p.add_argument("--spot", required=True)
    p.add_argument("--kind", choices=list(KINDS), default="all",
                   help="false-free = occupied predicted free; false-occupied = free predicted occupied")
    p.add_argument("--sort", choices=("confidence", "time"), default="confidence")
    p.add_argument("--n", type=int, default=60, help="errors per sheet")
    p.add_argument("--pages", type=int, default=1)
    p.add_argument("--outside", choices=("keep", "grey"), default="grey", help="as in the run")
    p.add_argument("--margin", type=float, default=0.0, help="crop_margin of the run")
    p.add_argument("--input-size", type=int, default=96)
    p.add_argument("--rois", type=Path, default=Path("label_studio/rois.json"))
    p.add_argument("--out", type=Path, default=Path("data/review"))
    args = p.parse_args()

    rows = [r for r in csv.DictReader(args.csv.open(encoding="utf-8")) if r["spot"] == args.spot]
    for r in rows:
        r["p_occupied"] = float(r["p_occupied"])
        r["hour"] = r["ts_utc"][11:13] + "h UTC"
        b, c = float(r["brightness"]), float(r["contrast"])
        r["quality"] = "dark" if b < 50 else "low_contrast" if c < 20 else "good"  # thresholds of train.py
    errors = [r for r in rows if r["label"] != r["pred"]]
    if KINDS[args.kind]:
        errors = [r for r in errors if (r["label"], r["pred"]) == KINDS[args.kind]]
    print(f"{args.spot}: {len(errors)} errors ({args.kind}) out of {len(rows)} crops "
          f"| labels {dict(Counter(r['label'] for r in rows))}")
    print(f"  types: {dict(Counter(f'{r['label']} -> {r['pred']}' for r in rows if r['label'] != r['pred']))}")
    for key, title in ((lambda r: r["day"], "Per day"), (lambda r: r["hour"], "Per hour"),
                       (lambda r: r["quality"], "Per image quality")):
        table(errors, key, title, Counter(key(r) for r in rows))
    episodes = Counter(r["episode_id"] for r in errors)
    print(f"\nErrors in {len(episodes)} episodes (runs of the same label); the 5 largest:")
    for ep, n in episodes.most_common(5):
        first = min((r for r in errors if r["episode_id"] == ep), key=lambda r: r["ts_utc"])
        print(f"  {n:3d} errors  episode {ep} from {first['ts_utc'][:16]}  ({first['label']})")
    if not errors:
        return

    key = (lambda r: -abs(r["p_occupied"] - 0.5)) if args.sort == "confidence" else (lambda r: r["ts_utc"])
    errors.sort(key=key)

    load_dotenv(Path(__file__).with_name(".env"))
    container = BlobServiceClient.from_connection_string(
        os.environ["AZURE_STORAGE_CONNECTION_STRING"]).get_container_client(os.getenv("PW_CONTAINER", "raw"))
    from azure_utils import latest_asset_file

    os.environ.setdefault("AZUREML_ARM_SUBSCRIPTION", "48ee8f92-9a0c-4ecd-b6ce-83845f85bf75")
    os.environ.setdefault("AZUREML_ARM_RESOURCEGROUP", "parkwatch")
    os.environ.setdefault("AZUREML_ARM_WORKSPACE_NAME", "parkwatch-workspace")
    content, _ = latest_asset_file("parkwatch-alignment", "alignment.csv")
    alignment = Alignment(list(csv.DictReader(io.StringIO(content.decode("utf-8")))))
    rois = json.loads(args.rois.read_text(encoding="utf-8"))
    spot = next(s for s in rois["spots"] if s["id"] == args.spot)
    polygon = spot_polygons(rois)[args.spot]

    def render(r: dict) -> tuple[Image.Image, Image.Image]:
        photo = Image.open(io.BytesIO(container.download_blob(r["blob"]).readall())).convert("RGB")
        warp = alignment.warps(r["blob"]).get(args.spot, IDENTITY)
        net = network_input(align_image(photo, warp), spot["points"], polygon, args.outside, args.margin, args.input_size)
        return net, context_view(photo, spot["points"], warp)

    args.out.mkdir(parents=True, exist_ok=True)
    for page in range(args.pages):
        chunk = errors[page * args.n:(page + 1) * args.n]
        if not chunk:
            break
        with ThreadPoolExecutor(8) as pool:
            images = list(pool.map(render, chunk))
        cols = 5  # pairs per row
        n_rows = (len(chunk) + cols - 1) // cols
        fig, axes = plt.subplots(n_rows, 2 * cols, figsize=(2 * cols * 1.9, n_rows * 2.1), squeeze=False,
                                 gridspec_kw={"width_ratios": [1, 1.5] * cols})
        for ax in axes.flat:
            ax.axis("off")
        for k, (r, (net, ctx)) in enumerate(zip(chunk, images)):
            a_net, a_ctx = axes[k // cols][2 * (k % cols)], axes[k // cols][2 * (k % cols) + 1]
            a_net.imshow(net, interpolation="nearest")
            a_ctx.imshow(ctx)
            a_net.set_title(f"{r['day'][5:]} {r['ts_utc'][11:16]}Z", fontsize=7)
            a_ctx.set_title(f"true {r['label']} / p(occ) {r['p_occupied']:.2f}", fontsize=7,
                            color="#c0392b" if abs(r["p_occupied"] - 0.5) > 0.4 else "#222")
        fig.suptitle(f"{args.spot} — {args.kind} errors, by {args.sort}, page {page + 1} "
                     f"({page * args.n + 1}-{page * args.n + len(chunk)} of {len(errors)})", fontsize=10)
        fig.tight_layout()
        out = args.out / f"{args.spot}_{args.kind}_{args.sort}_p{page + 1}.png"
        fig.savefig(out, dpi=100)
        plt.close(fig)
        print(f"-> {out}")


if __name__ == "__main__":
    main()
