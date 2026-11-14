"""ParkWatch — trains ParkNet (CNN from scratch) with day-grouped cross-validation.

Flow:
  1. Holdout days (--holdout-days) are kept out; the rest are development days.
  2. Cross-validation grouped by day (--folds): each fold is trained for --epochs epochs and
     its validation macro F1 is measured after every epoch.
  3. Number of epochs = the one with the best mean macro F1 across folds.
  4. Final model: trained on ALL development days up to that epoch, with the same learning
     rate schedule as in cross-validation.
  5. If there is a holdout, it is evaluated (macro F1 = promotion criterion) with breakdowns
     per spot and per image quality. Without holdout there is no promotion metric.

Everything is logged to MLflow. Meant for an Azure ML compute cluster: inputs and outputs
come as arguments, no repo paths. With --mlflow-url the run goes to the self-hosted MLflow
(ml/tracking.py, one run per training, completed later by promote.py); without it, or if
that server cannot be reached, to the Azure ML job's own run (MLFLOW_RUN_ID), as before.

Usage:
  python train.py --data-dir <dataset> --output-dir outputs
  python train.py --data-dir <dataset> --holdout-days 2026-10-19 2026-10-22 --jitter 0.05
  python train.py --data-dir <dataset> --mlflow-url https://<app> --keyvault-url https://<vault>
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from data import CLASSES, SpotCrops, channel_stats, read_manifest
from metrics import evaluate
from parknet import ParkNet, count_parameters
import tracking

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # Windows console (cp1252) vs MLflow emojis


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    p.add_argument("--holdout-days", nargs="*", default=[], help="YYYY-MM-DD; empty = no holdout yet")
    p.add_argument("--exclude-days", nargs="*", default=[], help="YYYY-MM-DD left out of training (experiments)")
    p.add_argument("--holdout-eval", default="true",
                   help="false = the holdout days are kept out of training but not evaluated (experiments: "
                        "they are compared on the cross-validation, the holdout is only used to promote)")
    p.add_argument("--note", default="", help="experiment name: MLflow tag `experiment` and suffix of the run name")
    p.add_argument("--folds", type=int, default=4)
    # model
    p.add_argument("--width", type=float, default=1.0, help="filter multiplier (1.0 = 32/64/128/128)")
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--input-size", type=int, default=96)
    # crop and augmentation
    p.add_argument("--crop-margin", type=float, default=0.10)
    p.add_argument("--jitter", type=float, default=0.0, help="+-fraction of crop shift/scale (training only)")
    p.add_argument("--no-color-aug", action="store_true")
    p.add_argument("--no-hflip", action="store_true")
    # training
    p.add_argument("--epochs", type=int, default=40, help="maximum; cross-validation picks how many to use")
    p.add_argument("--final-epochs", type=int, default=None, help="forces the epochs of the final model (skips the CV choice)")
    p.add_argument("--skip-cv", action="store_true", help="no cross-validation (requires --final-epochs)")
    p.add_argument("--run-cv", default="true",
                   help="false = epochs of the experiment's last run with cross-validation (if none, CV is run)")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    # analysis by image quality (provisional thresholds on the crop's 0-255 grey level)
    p.add_argument("--dark-thr", type=float, default=50.0)
    p.add_argument("--low-contrast-thr", type=float, default=20.0)
    # tracking
    p.add_argument("--experiment", default="parkwatch-training")
    p.add_argument("--mlflow-url", default=None, help="self-hosted MLflow; none = Azure ML (or local) MLflow")
    p.add_argument("--keyvault-url", default=None, help="Key Vault with the pipeline user's password")
    args = p.parse_args()
    if args.skip_cv and args.final_epochs is None:
        p.error("--skip-cv requires --final-epochs")
    return args


def is_false(value: str) -> bool:
    """Boolean flags from Azure ML: a literal may arrive as False/false/0."""
    return str(value).strip().lower() in ("false", "0", "no")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def day_folds(days: list[str], k: int) -> list[list[str]]:
    """Deterministic split of days into k interleaved folds (consecutive days land in different folds)."""
    days = sorted(days)
    k = max(2, min(k, len(days)))
    return [days[i::k] for i in range(k)]


def has_both_classes(rows: list[dict]) -> bool:
    return len({r["label"] for r in rows}) == len(CLASSES)


class Trainer:
    def __init__(self, args, device: str):
        self.args, self.device = args, device

    def dataset(self, rows, mean, std, train: bool) -> SpotCrops:
        a = self.args
        return SpotCrops(
            a.data_dir, rows, a.input_size, a.crop_margin, mean, std,
            augment=train, jitter=a.jitter, color_aug=not a.no_color_aug, hflip=not a.no_hflip,
        )

    def loader(self, ds: SpotCrops, shuffle: bool) -> DataLoader:
        a = self.args
        return DataLoader(
            ds, batch_size=a.batch_size, shuffle=shuffle, num_workers=a.num_workers,
            pin_memory=self.device == "cuda", persistent_workers=a.num_workers > 0,
        )

    def build(self, train_rows):
        """New model + class-weighted loss + optimiser and LR schedule."""
        a = self.args
        model = ParkNet(len(CLASSES), a.width, a.dropout).to(self.device)
        counts = Counter(r["label"] for r in train_rows)
        weights = torch.tensor(
            [len(train_rows) / (len(CLASSES) * max(1, counts[c])) for c in CLASSES], dtype=torch.float32, device=self.device
        )
        loss_fn = nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
        # T_max = --epochs always: the final model stops at the chosen epoch with the SAME schedule as in CV
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=a.epochs)
        return model, loss_fn, optimizer, scheduler, weights.tolist()

    def train_epoch(self, model, dl, loss_fn, optimizer) -> float:
        model.train()
        total = 0.0
        for x, y, _ in dl:
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            loss = loss_fn(model(x), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(y)
        return total / len(dl.dataset)

    @torch.no_grad()
    def predict(self, model, dl):
        model.eval()
        ys, ps, probs, idxs = [], [], [], []
        for x, y, idx in dl:
            p = torch.softmax(model(x.to(self.device, non_blocking=True)), 1).cpu()
            ps.append(p.argmax(1))
            probs.append(p[:, CLASSES.index("occupied")])
            ys.append(y)
            idxs.append(idx)
        return torch.cat(ys).numpy(), torch.cat(ps).numpy(), torch.cat(idxs).numpy(), torch.cat(probs).numpy()

    def evaluate(self, model, dl, ds: SpotCrops) -> dict:
        y, pred, idx, prob = self.predict(model, dl)
        m = evaluate(y, pred, [ds.rows[i] for i in idx], CLASSES, self.args.dark_thr, self.args.low_contrast_thr)
        m["_y"], m["_pred"], m["_idx"], m["_prob"] = y, pred, idx, prob
        return m


def flat_metrics(prefix: str, m: dict) -> dict:
    return {f"{prefix}/{k}": v for k, v in m.items() if not k.startswith("_") and k != "confusion_matrix"}


def cross_validate(trainer: Trainer, dev_rows, folds, mlflow) -> tuple[int, np.ndarray, list]:
    """Returns (chosen epochs, F1 curves per fold, validation predictions per fold and epoch)."""
    a = trainer.args
    curves = np.full((len(folds), a.epochs), np.nan)
    # oof[k] = (validation rows of fold k, [(y, pred, p_occupied) per epoch]) for the out-of-fold report
    oof: list[tuple[list[dict], list[tuple[np.ndarray, np.ndarray]]]] = []
    for k, val_days in enumerate(folds):
        tr = [r for r in dev_rows if r["day"] not in val_days]
        va = [r for r in dev_rows if r["day"] in val_days]
        if not has_both_classes(va):
            print(f"  warning: fold {k} ({val_days}) does not have both classes in validation; its macro F1 is biased")
        mean, std = channel_stats(a.data_dir, tr, a.input_size, a.crop_margin, seed=a.seed)
        tr_dl = trainer.loader(trainer.dataset(tr, mean, std, train=True), shuffle=True)
        va_ds = trainer.dataset(va, mean, std, train=False)
        va_dl = trainer.loader(va_ds, shuffle=False)
        model, loss_fn, opt, sched, _ = trainer.build(tr)
        print(f"fold {k}: val {val_days} ({len(va)} crops), train {len(tr)}")
        per_epoch = []
        for epoch in range(a.epochs):
            t0 = time.time()
            loss = trainer.train_epoch(model, tr_dl, loss_fn, opt)
            sched.step()
            m = trainer.evaluate(model, va_dl, va_ds)
            order = np.argsort(m["_idx"])  # aligned with va_ds.rows
            per_epoch.append((m["_y"][order], m["_pred"][order], m["_prob"][order].astype(np.float32)))
            curves[k, epoch] = m["f1_macro"]
            mlflow.log_metrics({f"cv/fold{k}/loss_train": loss, f"cv/fold{k}/f1_macro_val": m["f1_macro"]}, step=epoch)
            print(f"  epoch {epoch + 1:2d}: loss {loss:.3f} | val F1 {m['f1_macro']:.4f} ({time.time() - t0:.0f}s)")
        oof.append((va, per_epoch))
    mean_curve = np.nanmean(curves, axis=0)
    for epoch, v in enumerate(mean_curve):
        mlflow.log_metrics({"cv/f1_macro_mean": float(v), "cv/f1_macro_std": float(np.nanstd(curves[:, epoch]))}, step=epoch)
    best_epochs = int(np.nanargmax(mean_curve)) + 1
    return best_epochs, curves, oof


def log_oof_report(trainer: Trainer, oof, epochs: int, mlflow) -> dict:
    """Out-of-fold predictions at the chosen epoch: every development crop predicted once, by the
    model of the fold that did not see it. Confusion matrix, errors and per spot/quality breakdowns.

    Also every prediction (eval/cv_oof_predictions.csv, for tools/review_errors.py) and one figure
    of errors per spot, the most confident first (often wrong labels)."""
    a = trainer.args
    rows = [r for va, _ in oof for r in va]
    fold = np.concatenate([np.full(len(va), k) for k, (va, _) in enumerate(oof)])
    y = np.concatenate([per_epoch[epochs - 1][0] for _, per_epoch in oof])
    pred = np.concatenate([per_epoch[epochs - 1][1] for _, per_epoch in oof])
    prob = np.concatenate([per_epoch[epochs - 1][2] for _, per_epoch in oof])
    m = evaluate(y, pred, rows, CLASSES, a.dark_thr, a.low_contrast_thr)
    m["_y"], m["_pred"], m["_idx"], m["_prob"] = y, pred, np.arange(len(rows)), prob
    mlflow.log_metrics(flat_metrics("cv_oof", m))
    mlflow.log_figure(confusion_figure(m["confusion_matrix"], CLASSES), "eval/cv_oof_confusion.png")
    mlflow.log_text(predictions_csv(rows, y, pred, prob, fold), "eval/cv_oof_predictions.csv")
    ds = SpotCrops(a.data_dir, rows, a.input_size, a.crop_margin)
    fig = errors_figure(ds, m)
    if fig is not None:
        mlflow.log_figure(fig, "eval/cv_oof_errors.png")
    for spot in sorted({r["spot"] for r in rows}):
        fig = errors_figure(ds, m, max_n=60, spot=spot)
        if fig is not None:
            mlflow.log_figure(fig, f"eval/cv_oof_errors_{spot}.png")
    print(f"CV out-of-fold (epoch {epochs}): macro F1 {m['f1_macro']:.4f} | acc {m['acc']:.4f} | matrix {m['confusion_matrix']}")
    return m


PREDICTION_COLUMNS = ["blob", "day", "ts_utc", "spot", "label", "pred", "p_occupied", "fold",
                      "brightness", "contrast", "episode_id", "episode_len", "align_dx", "align_dy", "crop"]


def predictions_csv(rows: list[dict], y, pred, prob, fold) -> str:
    """One row per crop: label, prediction and probability of `occupied` (out-of-fold)."""
    import csv
    import io

    out = io.StringIO()
    w = csv.DictWriter(out, fieldnames=PREDICTION_COLUMNS, extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    for r, yi, pi, qi, fi in zip(rows, y, pred, prob, fold):
        w.writerow({**r, "label": CLASSES[yi], "pred": CLASSES[pi], "p_occupied": round(float(qi), 4), "fold": int(fi)})
    return out.getvalue()


def epochs_from_last_cv(mlflow) -> tuple[int, str] | None:
    """(epochs, run_id) of the latest finished run of this experiment that ran cross-validation."""
    # The Azure ML search filter does not accept metric names containing "/" (cv/best_epochs):
    # the experiment's runs are fetched and filtered here.
    run = mlflow.active_run()
    runs = mlflow.search_runs(experiment_ids=[run.info.experiment_id], max_results=1000)
    column = "metrics.cv/best_epochs"
    if runs.empty or column not in runs.columns:
        return None
    done = runs[(runs["status"] == "FINISHED") & runs[column].notna() & (runs[column] > 0)]
    if done.empty:
        return None
    last = done.sort_values("start_time", ascending=False).iloc[0]
    return int(last[column]), last["run_id"]


def confusion_figure(cm, classes):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes)
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i][j], ha="center", va="center")
    fig.tight_layout()
    return fig


def errors_figure(ds: SpotCrops, m: dict, max_n: int = 36, spot: str | None = None):
    """Wrong crops as the network sees them; with `spot`, only that spot and the most confident first."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wrong = [k for k in range(len(m["_y"])) if m["_y"][k] != m["_pred"][k]]
    if spot is not None:
        wrong = [k for k in wrong if ds.rows[m["_idx"][k]]["spot"] == spot]
        wrong.sort(key=lambda k: -abs(float(m["_prob"][k]) - 0.5))
    wrong = wrong[:max_n]
    if not wrong:
        return None
    cols = 6
    rows = (len(wrong) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.5), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, k in zip(axes.flat, wrong):
        i = m["_idx"][k]
        row = ds.rows[i]
        ax.imshow(ds.load_image(i))
        ax.set_title(
            f"{row['spot']} {row['day'][5:]} {row['ts_utc'][11:16]}Z\n"
            f"true {CLASSES[m['_y'][k]]} / pred {CLASSES[m['_pred'][k]]}"
            + (f" ({float(m['_prob'][k]):.2f})" if "_prob" in m else ""),
            fontsize=7,
        )
    fig.tight_layout()
    return fig


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = device == "cuda"

    rows = read_manifest(args.data_dir)
    info = json.loads((args.data_dir / "dataset_info.json").read_text(encoding="utf-8"))
    holdout_days = set(args.holdout_days)
    unknown = holdout_days - {r["day"] for r in rows}
    if unknown:
        # a day reserved for the holdout before it is labelled (so that no model ever sees it)
        print(f"warning: holdout days without labels in the dataset yet: {sorted(unknown)}")
    excluded = set(args.exclude_days)  # experiments: days left out of training (and not evaluated)
    dev = [r for r in rows if r["day"] not in holdout_days and r["day"] not in excluded]
    if excluded:
        print(f"Excluded from training: {sorted(excluded)}")
    holdout = [r for r in rows if r["day"] in holdout_days]
    dev_days = sorted({r["day"] for r in dev})
    folds = day_folds(dev_days, args.folds)
    trainer = Trainer(args, device)

    import mlflow

    self_hosted = tracking.use_self_hosted(args.mlflow_url, args.keyvault_url, args.experiment)
    # Inside an Azure ML job tracking to the workspace, the run already exists (MLFLOW_RUN_ID).
    job_run = "MLFLOW_RUN_ID" in os.environ
    if not job_run:
        mlflow.set_experiment(args.experiment)
    name = tracking.run_name(vars(args)) + (f"_{args.note}" if args.note else "")
    with mlflow.start_run(run_name=None if job_run else name) as run:
        if args.note:
            mlflow.set_tag("experiment", args.note)
        if self_hosted and tracking.AZUREML_RUN_ID:
            mlflow.set_tag(tracking.TAG_AZURE_TRAIN_RUN, tracking.AZUREML_RUN_ID)
        tracking.link_azureml_run(run)
        n_params = count_parameters(ParkNet(len(CLASSES), args.width, args.dropout))
        mlflow.log_params(
            {
                # final_epochs is logged at the end with the value used (MLflow does not allow changing a param)
                **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k not in ("holdout_days", "final_epochs", "mlflow_url", "keyvault_url")},
                "final_epochs_forced": args.final_epochs or "none",
                "arch": "parknet",
                "n_params": n_params,
                "device": device,
                "rois_version": info["rois_version"],
                "dataset_built_utc": info["built_utc"],
                "dataset_fingerprint": info.get("fingerprint", "none"),
                "dev_days": ",".join(dev_days),
                "holdout_days": ",".join(sorted(holdout_days)) or "none",
                "cv_folds": json.dumps(folds),
                "n_dev": len(dev),
                "n_holdout": len(holdout),
            }
        )
        mlflow.log_dict(info, "dataset/dataset_info.json")
        print(f"ParkNet {n_params:,} parameters | {device} | dev {len(dev)} crops over {len(dev_days)} days | holdout {len(holdout)}")

        # 1-3. Day-grouped cross-validation -> number of epochs
        last_cv = None
        if args.final_epochs and is_false(args.run_cv):
            args.skip_cv = True  # forced epochs and no CV asked: nothing to look up
        if not args.skip_cv and is_false(args.run_cv):
            last_cv = epochs_from_last_cv(mlflow)
            if last_cv is None:
                print("No previous runs with cross-validation: running CV")
        if args.skip_cv:
            best_epochs = args.final_epochs
        elif last_cv is not None:
            best_epochs = last_cv[0]
            mlflow.log_param("epochs_from_run", last_cv[1])
            print(f"Epochs of the last run with CV ({last_cv[1]}): {best_epochs}")
        else:
            best_epochs, curves, oof = cross_validate(trainer, dev, folds, mlflow)
            best = curves[:, best_epochs - 1]
            mlflow.log_metrics({"cv/best_epochs": best_epochs, "cv/best_f1_mean": float(np.nanmean(best)), "cv/best_f1_std": float(np.nanstd(best))})
            print(f"CV: best epoch {best_epochs} -> macro F1 {np.nanmean(best):.4f} +- {np.nanstd(best):.4f}")
            if args.final_epochs:
                best_epochs = args.final_epochs
            log_oof_report(trainer, oof, min(best_epochs, args.epochs), mlflow)
        mlflow.log_param("final_epochs", best_epochs)

        # 4. Final model on all development days
        mean, std = channel_stats(args.data_dir, dev, args.input_size, args.crop_margin, seed=args.seed)
        dl = trainer.loader(trainer.dataset(dev, mean, std, train=True), shuffle=True)
        model, loss_fn, opt, sched, class_weights = trainer.build(dev)
        mlflow.log_params({"norm_mean": list(mean), "norm_std": list(std), "class_weights": [round(w, 3) for w in class_weights]})
        print(f"Final model: {best_epochs} epochs on {len(dev)} crops")
        for epoch in range(best_epochs):
            loss = trainer.train_epoch(model, dl, loss_fn, opt)
            sched.step()
            mlflow.log_metric("final/loss_train", loss, step=epoch)

        # 5. Holdout (if any)
        if holdout and is_false(args.holdout_eval):
            print(f"Holdout ({len(holdout)} crops) kept out but not evaluated (experiment: compare on the CV)")
        elif holdout:
            ho_ds = trainer.dataset(holdout, mean, std, train=False)
            m = trainer.evaluate(model, trainer.loader(ho_ds, shuffle=False), ho_ds)
            mlflow.log_metrics(flat_metrics("holdout", m))
            mlflow.log_figure(confusion_figure(m["confusion_matrix"], CLASSES), "eval/holdout_confusion.png")
            fig = errors_figure(ho_ds, m)
            if fig is not None:
                mlflow.log_figure(fig, "eval/holdout_errors.png")
            print(f"Holdout: macro F1 {m['f1_macro']:.4f} | acc {m['acc']:.4f}")
        else:
            print("No holdout: no promotion metric")

        # Artifacts: weights + everything needed to reproduce the preprocessing at inference time
        model_dir = args.output_dir / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_dir / "parknet.pt")
        config = {
            "arch": "parknet",
            "classes": list(CLASSES),
            "width": args.width,
            "dropout": args.dropout,
            "input_size": args.input_size,
            "crop_margin": args.crop_margin,
            "norm_mean": list(mean),
            "norm_std": list(std),
            "rois_version": info["rois_version"],
            "trained_days": dev_days,
            "epochs": best_epochs,
            # traceability for the model registry and for promote.py, which completes this run
            "mlflow_run_id": run.info.run_id,
            "mlflow_tracking": args.mlflow_url.rstrip("/") if self_hosted else "default",
            "azureml_run_id": tracking.AZUREML_RUN_ID or "none",
            "dataset_fingerprint": info.get("fingerprint", "none"),
        }
        (model_dir / "model_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        mlflow.log_artifacts(str(model_dir), "model")
        print(f"Model saved to {model_dir}")


if __name__ == "__main__":
    main()
