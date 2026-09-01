"""Evaluation metrics with no simulator dependency."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch


def regression_metrics(prediction: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    error = prediction - target
    return {"rmse": float(error.square().mean().sqrt()), "mae": float(error.abs().mean())}


def binary_metrics(probability: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    prediction = probability >= threshold
    target = target.bool()
    tp = (prediction & target).sum().float()
    fp = (prediction & ~target).sum().float()
    fn = (~prediction & target).sum().float()
    precision = tp / (tp + fp).clamp(min=1)
    recall = tp / (tp + fn).clamp(min=1)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-8)
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def uncertainty_error_correlation(uncertainty: torch.Tensor, squared_error: torch.Tensor) -> float:
    x = uncertainty.detach().cpu().numpy().reshape(-1)
    y = squared_error.detach().cpu().numpy().reshape(-1)
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def reward_ranking_metrics(
    prediction: torch.Tensor, target: torch.Tensor, top_k: int = 5
) -> Dict[str, float]:
    """Measure whether a model orders candidates as the simulator does."""

    p = prediction.detach().cpu().double().reshape(-1).numpy()
    t = target.detach().cpu().double().reshape(-1).numpy()
    finite = np.isfinite(p) & np.isfinite(t)
    p, t = p[finite], t[finite]
    if len(p) < 2:
        return {key: float("nan") for key in (
            "pearson", "spearman", "pairwise_accuracy", "top_k_overlap",
            "selection_regret",
        )}
    pearson = float(np.corrcoef(p, t)[0, 1]) if np.std(p) and np.std(t) else float("nan")
    # Average ranks preserve ties, which are common in sparse football reward.
    def ranks(values):
        order = np.argsort(values, kind="stable")
        result = np.empty(len(values), dtype=np.float64)
        start = 0
        while start < len(values):
            stop = start + 1
            while stop < len(values) and values[order[stop]] == values[order[start]]:
                stop += 1
            result[order[start:stop]] = 0.5 * (start + stop - 1)
            start = stop
        return result
    pr, tr = ranks(p), ranks(t)
    spearman = float(np.corrcoef(pr, tr)[0, 1]) if np.std(pr) and np.std(tr) else float("nan")
    left, right = np.triu_indices(len(p), k=1)
    comparable = t[left] != t[right]
    predicted_difference = p[left] - p[right]
    target_difference = t[left] - t[right]
    if comparable.any():
        pairwise = np.where(
            predicted_difference[comparable] == 0.0,
            0.5,
            (np.sign(predicted_difference[comparable]) == np.sign(target_difference[comparable])).astype(float),
        ).mean()
    else:
        pairwise = float("nan")
    k = max(1, min(int(top_k), len(p)))
    predicted_top = set(np.argpartition(p, -k)[-k:].tolist())
    actual_top = set(np.argpartition(t, -k)[-k:].tolist())
    return {
        "pearson": pearson,
        "spearman": spearman,
        "pairwise_accuracy": float(pairwise),
        "top_k_overlap": len(predicted_top & actual_top) / k,
        "selection_regret": float(t.max() - t[int(np.argmax(p))]),
    }
