"""ParkWatch — evaluates several models on the same holdout (diagnostics: nothing is registered or promoted).

Models are given as train outputs (--model-dirs) and/or registered versions of parknet
(--versions). Each one crops with its own config (model_config.json), exactly like promote.py.

  python evaluate_models.py --data-dir <dataset> --holdout-days 2026-10-23 2026-11-03 \
      --versions 11 15 --model-dirs repro=<train output>
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from data import read_manifest
from promote import holdout_metrics
from registry import MODEL_NAME, download, model_root


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, required=True, help="build_dataset output")
    p.add_argument("--holdout-days", nargs="+", required=True)
    p.add_argument("--versions", nargs="*", default=[], help="registered parknet versions")
    p.add_argument("--model-dirs", nargs="*", default=[], help="name=path of train outputs")
    p.add_argument("--dark-thr", type=float, default=50.0)
    p.add_argument("--low-contrast-thr", type=float, default=20.0)
    p.add_argument("--num-workers", type=int, default=2)
    args = p.parse_args()

    rows = [r for r in read_manifest(args.data_dir) if r["day"] in set(args.holdout_days)]
    days = sorted({r["day"] for r in rows})
    print(f"Holdout: {len(rows)} crops over {len(days)} days {days}")
    models = {name: model_root(Path(path)) for name, path in (m.split("=", 1) for m in args.model_dirs)}
    tmp = Path(tempfile.mkdtemp())

    class Version:  # what registry.download needs
        def __init__(self, v: str):
            self.version = v

    for v in args.versions:
        models[f"{MODEL_NAME}:{v}"] = download(Version(v), tmp / v)

    results = {name: holdout_metrics(path, args.data_dir, rows, args) for name, path in models.items()}
    keys = ["f1_macro", "acc"] + [f"spot/{s}/acc" for s in sorted({r["spot"] for r in rows})] \
        + [f"quality/{q}/acc" for q in ("good", "low_contrast", "dark")]

    print(f"{'':24s}" + "".join(f"{name:>20s}" for name in results))
    for key in keys:
        cells = [m.get(key) for m in results.values()]
        print(f"{key:24s}" + "".join(f"{c:20.4f}" if c is not None else f"{'-':>20s}" for c in cells))


if __name__ == "__main__":
    main()
