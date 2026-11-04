"""Evaluation metrics: macro F1 (the decision criterion) + informative breakdowns."""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, classes: tuple[str, ...]) -> dict:
    n = len(classes)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    out = {"acc": float((y_true == y_pred).mean()) if len(y_true) else 0.0, "n": int(len(y_true))}
    f1s = []
    for i, c in enumerate(classes):
        tp = cm[i, i]
        precision = tp / cm[:, i].sum() if cm[:, i].sum() else 0.0
        recall = tp / cm[i, :].sum() if cm[i, :].sum() else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out[f"precision/{c}"], out[f"recall/{c}"], out[f"f1/{c}"] = float(precision), float(recall), float(f1)
        f1s.append(f1)
    out["f1_macro"] = float(np.mean(f1s))
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
            m = classification_metrics(y_true[idx], y_pred[idx], classes)
            out[f"{kind}/{name}/acc"] = m["acc"]
            out[f"{kind}/{name}/f1_macro"] = m["f1_macro"]
            out[f"{kind}/{name}/n"] = m["n"]
    return out
