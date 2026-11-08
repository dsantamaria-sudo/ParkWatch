"""Evaluation metrics: macro F1 (the decision criterion) + informative breakdowns."""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, classes: tuple[str, ...],
                           present_only: bool = False) -> dict:
    """Accuracy, per-class precision/recall/F1, macro F1 and confusion matrix.

    present_only: the macro F1 only averages the classes that appear in the labels or in the
    predictions. Used for the per-spot / per-quality breakdowns: a spot that is always occupied
    in the holdout would otherwise get F1 0 for `free` and a macro F1 of 0.5 even when perfect."""
    n = len(classes)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    out = {"acc": float((y_true == y_pred).mean()) if len(y_true) else 0.0, "n": int(len(y_true))}
    f1s = []
    present = set(np.asarray(y_true).tolist()) | set(np.asarray(y_pred).tolist())
    for i, c in enumerate(classes):
        tp = cm[i, i]
        precision = tp / cm[:, i].sum() if cm[:, i].sum() else 0.0
        recall = tp / cm[i, :].sum() if cm[i, :].sum() else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out[f"precision/{c}"], out[f"recall/{c}"], out[f"f1/{c}"] = float(precision), float(recall), float(f1)
        if not present_only or i in present:
            f1s.append(f1)
    out["f1_macro"] = float(np.mean(f1s)) if f1s else 0.0
    out["confusion_matrix"] = cm.tolist()
    return out


def quality_bucket(row: dict, dark_thr: float, low_contrast_thr: float) -> str:
    if row["brightness"] < dark_thr:
        return "dark"
    if row["contrast"] < low_contrast_thr:
        return "low_contrast"
    return "good"


def evaluate(y_true, y_pred, rows: list[dict], classes, dark_thr: float, low_contrast_thr: float) -> dict:
    """Global metrics + per spot + per image quality (flat keys ready for MLflow)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    out = classification_metrics(y_true, y_pred, classes)
    groups = {"spot": defaultdict(list), "quality": defaultdict(list)}
    for k, row in enumerate(rows):
        groups["spot"][row["spot"]].append(k)
        groups["quality"][quality_bucket(row, dark_thr, low_contrast_thr)].append(k)
    for kind, buckets in groups.items():
        for name, idx in sorted(buckets.items()):
            m = classification_metrics(y_true[idx], y_pred[idx], classes, present_only=True)
            out[f"{kind}/{name}/acc"] = m["acc"]
            out[f"{kind}/{name}/f1_macro"] = m["f1_macro"]  # classes present only (see above)
            out[f"{kind}/{name}/n"] = m["n"]
            for i, c in enumerate(classes):  # how many of each class: a spot with 3 free crops says little
                out[f"{kind}/{name}/n_{c}"] = int((y_true[idx] == i).sum())
    return out
