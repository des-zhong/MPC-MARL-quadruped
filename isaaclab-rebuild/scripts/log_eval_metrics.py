"""Append periodic evaluation JSON metrics to a TensorBoard run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--metrics", type=Path, required=True)
parser.add_argument("--log_dir", type=Path, required=True)
parser.add_argument("--iteration", type=int, required=True)
args = parser.parse_args()

payload = json.loads(args.metrics.read_text(encoding="utf-8"))
skill_counts = payload.get("skill_counts", {})
total_skills = max(sum(int(value) for value in skill_counts.values()), 1)
with SummaryWriter(log_dir=str(args.log_dir)) as writer:
    writer.add_scalar("Eval/mean_return", float(payload["mean_return"]), args.iteration)
    writer.add_scalar("Eval/mean_length", float(payload["mean_length"]), args.iteration)
    writer.add_scalar("Eval/completed_episodes", int(payload["completed_episodes"]), args.iteration)
    for name in ("walk", "dribble", "shoot"):
        writer.add_scalar(
            f"Eval/skill_fraction/{name}",
            int(skill_counts.get(name, 0)) / total_skills,
            args.iteration,
        )

print(f"[EVAL] TensorBoard metrics appended at iteration {args.iteration}")
