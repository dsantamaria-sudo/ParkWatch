"""ParkWatch — per-day and per-spot statistics to choose the holdout days.

Reads manifest.csv of the latest version (or --version) of the parkwatch-crops data asset in
the Azure ML workspace, without downloading the crops, and shows per day:
  - free / total crops and free episodes of each spot
  - dark and low-contrast crops (same thresholds as train.py)
With --holdout, the resulting split: how much goes to holdout and to train per spot.

Usage (repo root, with az login):
  uv run python tools/holdout_stats.py
  uv run python tools/holdout_stats.py --holdout 2026-10-19 2026-10-22
"""
from __future__ import annotations

import argparse
import csv
import io
from collections import Counter, defaultdict

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobClient

SUBSCRIPTION = "48ee8f92-9a0c-4ecd-b6ce-83845f85bf75"
RESOURCE_GROUP = "parkwatch"
WORKSPACE = "parkwatch-workspace"


def load_manifest(asset: str, version: str | None) -> tuple[list[dict], str]:
    cred = DefaultAzureCredential()
    client = MLClient(cred, SUBSCRIPTION, RESOURCE_GROUP, WORKSPACE)
    data = client.data.get(asset, version=version) if version else client.data.get(asset, label="latest")
    datastore_name, rel = data.path.split("/datastores/", 1)[1].split("/paths/", 1)
    ds = client.datastores.get(datastore_name)
    blob = BlobClient(f"https://{ds.account_name}.blob.core.windows.net", ds.container_name,
                      f"{rel.rstrip('/')}/manifest.csv", credential=cred)
    text = blob.download_blob().readall().decode("utf-8")
    return list(csv.DictReader(io.StringIO(text))), data.version


def summarize(rows: list[dict], dark_thr: float, low_contrast_thr: float):
    spots = sorted({r["spot"] for r in rows})
    days = sorted({r["day"] for r in rows})
    free = Counter((r["day"], r["spot"]) for r in rows if r["label"] == "free")
    total = Counter((r["day"], r["spot"]) for r in rows)
    episodes = defaultdict(set)
    for r in rows:
        if r["label"] == "free":
            episodes[(r["day"], r["spot"])].add(r["episode_id"])
    dark = Counter(r["day"] for r in rows if float(r["brightness"]) < dark_thr)
    lowc = Counter(r["day"] for r in rows if float(r["brightness"]) >= dark_thr and float(r["contrast"]) < low_contrast_thr)
    per_day = Counter(r["day"] for r in rows)
    return spots, days, free, total, episodes, dark, lowc, per_day


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--asset", default="parkwatch-crops")
    p.add_argument("--version", default=None)
    p.add_argument("--holdout", nargs="*", default=[], help="candidate days YYYY-MM-DD")
    p.add_argument("--dark-thr", type=float, default=50.0)
    p.add_argument("--low-contrast-thr", type=float, default=20.0)
    args = p.parse_args()

    rows, version = load_manifest(args.asset, args.version)
    spots, days, free, total, episodes, dark, lowc, per_day = summarize(rows, args.dark_thr, args.low_contrast_thr)
    print(f"{args.asset}:{version} — {len(rows)} crops, {len(days)} days\n")
    print("Free per day and spot: free/total crops (free episodes)")
    header = f"{'day':10}  " + "  ".join(f"{s:>15}" for s in spots) + f"  {'crops':>8}  {'dark':>7}  {'low_contr':>10}"
    print(header)
    for d in days:
        cells = "  ".join(f"{free[(d, s)]:>5}/{total[(d, s)]:<4} ({len(episodes[(d, s)]):>2})" for s in spots)
        print(f"{d:10}  {cells}  {per_day[d]:>8}  {dark[d]:>7}  {lowc[d]:>10}")

    if args.holdout:
        unknown = set(args.holdout) - set(days)
        if unknown:
            raise SystemExit(f"days without data: {sorted(unknown)}")
        ho = set(args.holdout)
        n_ho = sum(per_day[d] for d in ho)
        print(f"\nHoldout {sorted(ho)}: {n_ho} crops ({n_ho / len(rows):.0%})")
        print(f"{'spot':6}  {'free in holdout (ep.)':>21}  {'free in train (ep.)':>19}")
        for s in spots:
            fh = sum(free[(d, s)] for d in ho)
            eh = sum(len(episodes[(d, s)]) for d in ho)
            ft = sum(free[(d, s)] for d in days if d not in ho)
            et = sum(len(episodes[(d, s)]) for d in days if d not in ho)
            print(f"{s:6}  {fh:>14} ({eh:>3})  {ft:>12} ({et:>3})")


if __name__ == "__main__":
    main()
