"""Decompose terminal-state model error from terminal-value prediction error."""

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

from dribblebot.mpc.teacher_dataset import TeacherDataset
from dribblebot.mpc.terminal_value import compute_discounted_returns, load_value_checkpoint


@torch.no_grad()
def main(args):
    dataset = TeacherDataset(args.dataset)
    value, payload = load_value_checkpoint(args.checkpoint, args.device)
    rows = []
    for episode_index in range(len(dataset)):
        arrays = dataset.load_episode(episode_index).arrays
        returns = arrays.get("real_return_to_go")
        if returns is None or np.asarray(returns).dtype.kind not in "fiu":
            returns = compute_discounted_returns(
                arrays["real_reward"], arrays["terminated"], arrays["truncated"], payload["gamma"]
            )
        for step in range(len(arrays["step_id"])):
            predicted_plan = arrays["predicted_plan_states"][step]
            horizon = len(predicted_plan) - 1
            actual_step = step + horizon
            if actual_step >= len(arrays["step_id"]):
                continue
            predicted_state = torch.as_tensor(predicted_plan[-1], device=args.device).float()[None]
            actual_state = torch.as_tensor(arrays["global_state"][actual_step], device=args.device).float()[None]
            predicted_value = float(value.predict(predicted_state)[0])
            actual_value = float(value.predict(actual_state)[0])
            realized = float(returns[actual_step])
            rows.append({
                "episode_id": int(arrays["episode_id"][step]), "step_id": step,
                "horizon": horizon,
                "world_model_terminal_state_rmse": float(torch.mean((predicted_state - actual_state) ** 2).sqrt()),
                "value_error_on_actual_state": actual_value - realized,
                "value_difference_from_state_prediction_error": predicted_value - actual_value,
                "predicted_terminal_value": predicted_value,
                "actual_terminal_state_value": actual_value,
                "realized_continuation_return": realized,
            })
    if not rows:
        raise ValueError("No teacher rows have enough realized future steps for decomposition")
    aggregate = {
        key: {"mean": float(np.mean([row[key] for row in rows])),
              "rmse": float(np.sqrt(np.mean(np.square([row[key] for row in rows]))))}
        for key in ("world_model_terminal_state_rmse", "value_error_on_actual_state",
                    "value_difference_from_state_prediction_error")
    }
    result = {"samples": len(rows), "aggregate": aggregate, "rows": rows}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps({"samples": len(rows), "aggregate": aggregate}, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/mpc_teacher")
    parser.add_argument("--checkpoint", default="checkpoints/terminal_value/best.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="outputs/terminal_value_error_decomposition.json")
    main(parser.parse_args())
