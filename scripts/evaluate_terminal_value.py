"""Evaluate and calibrate terminal value predictions on held-out episodes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from dribblebot.mpc.terminal_value import ValueDataset, load_value_checkpoint, value_metrics


@torch.no_grad()
def main(args):
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    model, checkpoint = load_value_checkpoint(args.checkpoint, args.device)
    dataset = ValueDataset(args.dataset, args.split)
    predictions, targets = [], []
    for batch in DataLoader(dataset, batch_size=args.batch_size):
        predictions.append(model.predict(batch["global_state"].to(args.device)).cpu())
        targets.append(batch["return_to_go"])
    prediction, target = torch.cat(predictions), torch.cat(targets)
    metrics = value_metrics(prediction, target)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    np.savez_compressed(output / "predictions.npz", prediction=prediction.numpy(), target=target.numpy())
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(10, 9))
        axes[0, 0].scatter(target, prediction, s=5, alpha=.25)
        low = min(float(target.min()), float(prediction.min())); high = max(float(target.max()), float(prediction.max()))
        axes[0, 0].plot([low, high], [low, high], "k--"); axes[0, 0].set(xlabel="Actual return", ylabel="Predicted V")
        order = torch.argsort(target); bins = torch.chunk(order, min(10, len(order)))
        axes[0, 1].plot([float(target[b].mean()) for b in bins], [float(prediction[b].mean()) for b in bins], "o-")
        axes[0, 1].plot([low, high], [low, high], "k--"); axes[0, 1].set_title("Binned calibration")
        axes[1, 0].hist(target.numpy(), bins=50, alpha=.6, label="target"); axes[1, 0].hist(prediction.numpy(), bins=50, alpha=.6, label="prediction"); axes[1, 0].legend()
        axes[1, 1].hist((prediction - target).numpy(), bins=50); axes[1, 1].set_title("Residual")
        fig.tight_layout(); fig.savefig(output / "calibration.png", dpi=160); plt.close(fig)
    except ImportError:
        pass
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/terminal_value/best.pt")
    parser.add_argument("--dataset", default="data/terminal_value")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", default="outputs/terminal_value_evaluation")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=2048)
    main(parser.parse_args())
