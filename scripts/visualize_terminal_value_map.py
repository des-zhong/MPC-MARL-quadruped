"""Plot V(s) over valid unpossessed ball positions for one recorded state."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import torch

from dribblebot.mpc.terminal_value import ValueDataset, load_value_checkpoint


@torch.no_grad()
def main(args):
    model, _ = load_value_checkpoint(args.checkpoint, args.device)
    recorded = ValueDataset(args.dataset, args.split)[args.sample]["global_state"]
    axis = torch.linspace(-1, 1, args.resolution)
    x, y = torch.meshgrid(axis, axis, indexing="xy")
    states = recorded[None].expand(x.numel(), -1).clone()
    ball = model.schema.slice("ball.position")
    states[:, ball.start] = x.flatten(); states[:, ball.start + 1] = y.flatten()
    states[:, model.schema.slice("ball.possessed")] = 0
    possessor = model.schema.slice("ball.possessor_one_hot")
    states[:, possessor] = 0; states[:, possessor.start] = 1
    values = model.predict(states.to(args.device)).cpu().reshape(args.resolution, args.resolution)
    field = recorded[model.schema.slice("field.geometry")]
    extent = [-float(field[0]), float(field[0]), -float(field[1]), float(field[1])]
    figure, axis_plot = plt.subplots(figsize=(10, 6))
    image = axis_plot.imshow(values.numpy(), origin="lower", extent=extent, aspect="equal", cmap="viridis")
    figure.colorbar(image, ax=axis_plot, label="Predicted continuation return V(s)")
    axis_plot.set(title="Terminal value vs unpossessed ball position", xlabel="field x (m)", ylabel="field y (m)")
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(); figure.savefig(output, dpi=180); plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/terminal_value/best.pt")
    parser.add_argument("--dataset", default="data/terminal_value")
    parser.add_argument("--split", default="test")
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=80)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="outputs/terminal_value_ball_heatmap.png")
    main(parser.parse_args())
