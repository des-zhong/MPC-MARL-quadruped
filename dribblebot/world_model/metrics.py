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
