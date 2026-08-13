"""Evaluate one-step, multi-step, and uncertainty calibration metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dribblebot.world_model.dataset import WorldModelDataset
from dribblebot.world_model.metrics import binary_metrics, regression_metrics, uncertainty_error_correlation
from dribblebot.world_model.schema import event_names_from_metadata, validate_event_names
from dribblebot.world_model.trainer import load_checkpoint


def _group_metrics(model, prediction, target):
    result = {}
    for group in ("robot_position", "robot_velocity", "ball_position", "ball_velocity"):
        indices = [i for f in model.schema.features if f.group == group for i in range(f.start, f.stop)]
        if indices:
            result[group] = regression_metrics(prediction[:, indices], target[:, indices])
    return result


def _physical_position_rmse(model, prediction, target, group):
    field = target[:, model.schema.slice("field.geometry")]
    scales = torch.stack((field[:, 0], field[:, 1], torch.ones_like(field[:, 0])), -1)
    errors = []
    for feature in model.schema.features:
        if feature.group == group and feature.name.endswith("position"):
            errors.append((prediction[:, feature.start : feature.stop] - target[:, feature.start : feature.stop]) * scales)
    return float(torch.cat(errors, dim=-1).square().mean().sqrt()) if errors else float("nan")


@torch.no_grad()
def one_step(model, dataset, device, batch_size, event_names):
    predictions, targets, rewards, reward_targets, dones, done_targets, event_probs, event_targets = [], [], [], [], [], [], [], []
    uncertainty, squared_error = [], []
    nll_values = []
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        state = batch["state"].float().to(device)
        action = batch["joint_action"].float().to(device)
        target = batch["next_state"].float().to(device)
        next_state, reward, done, events, unc = model.predict_next(state, action)
        predictions.append(next_state.cpu()); targets.append(target.cpu())
        rewards.append(reward.cpu()); reward_targets.append(batch["reward"].float())
        dones.append(done.cpu()); done_targets.append((batch["terminated"].bool() | batch["truncated"].bool()).float())
        event_probs.append(events.cpu()); event_targets.append(batch["event_labels"].float())
        dynamic_error = (next_state[:, model.schema.continuous_dynamic_indices] - target[:, model.schema.continuous_dynamic_indices]).square().mean(-1)
        uncertainty.append(unc["mean_state_uncertainty"].cpu()); squared_error.append(dynamic_error.cpu())
        outputs = model.forward_members(state, action)
        delta_target = target[:, model.schema.continuous_dynamic_indices] - state[:, model.schema.continuous_dynamic_indices]
        member_mean = model.normalizer.denormalize_delta_prediction(outputs["delta_mean"])
        member_var = outputs["delta_log_variance"].exp() * model.normalizer.delta_std.square()
        nll_values.append((0.5 * ((delta_target[None] - member_mean).square() / member_var + member_var.log())).mean().cpu())
    prediction, target = torch.cat(predictions), torch.cat(targets)
    reward, reward_target = torch.cat(rewards), torch.cat(reward_targets)
    done, done_target = torch.cat(dones), torch.cat(done_targets)
    event, event_target = torch.cat(event_probs), torch.cat(event_targets)
    calibration = []
    for lower in torch.linspace(0, 0.9, 10):
        selected = (done >= lower) & (done < lower + 0.1)
        calibration.append({
            "predicted": float(done[selected].mean()) if selected.any() else None,
            "observed": float(done_target[selected].mean()) if selected.any() else None,
            "count": int(selected.sum()),
        })
    normalized_error = (prediction - target) / model.normalizer.state_std.cpu()
    uncertainty_all = torch.cat(uncertainty)
    squared_error_all = torch.cat(squared_error)
    quantiles = torch.quantile(uncertainty_all, torch.linspace(0, 1, 6))
    quantile_errors = []
    for index in range(5):
        selected = (uncertainty_all >= quantiles[index]) & (uncertainty_all <= quantiles[index + 1])
        quantile_errors.append(float(squared_error_all[selected].mean()) if selected.any() else float("nan"))
    groups = _group_metrics(model, prediction, target)
    groups["robot_position"]["physical_rmse_m"] = _physical_position_rmse(model, prediction, target, "robot_position")
    groups["ball_position"]["physical_rmse_m"] = _physical_position_rmse(model, prediction, target, "ball_position")
    return {
        "state_encoded_units": regression_metrics(prediction, target),
        "state_normalized": {"rmse": float(normalized_error.square().mean().sqrt()), "mae": float(normalized_error.abs().mean())},
        "groups": groups,
        "reward": regression_metrics(reward, reward_target),
        "termination": {**binary_metrics(done, done_target), "reliability": calibration},
        "events_micro": binary_metrics(event, event_target),
        "events": {name: binary_metrics(event[:, index], event_target[:, index]) for index, name in enumerate(event_names)},
        "state_nll": float(torch.stack(nll_values).mean()),
        "uncertainty": {
            "squared_error_correlation": uncertainty_error_correlation(uncertainty_all, squared_error_all),
            "error_by_uncertainty_quintile": quantile_errors,
            "in_distribution_mean_squared_error": float(squared_error_all.mean()),
            "unusual_state_error": None,
        },
    }


@torch.no_grad()
def multi_step(model, dataset, device, horizons):
    results = {}
    for horizon in horizons:
        available = dataset.sequences(horizon)
        if not available:
            results[str(horizon)] = {"count": 0}
            continue
        errors = {"robot_position": [], "ball_position": [], "ball_velocity": [], "cumulative_reward": [], "termination_brier": []}
        for episode_index, start in available[:1000]:
            sequence = dataset.get_sequence(episode_index, start, horizon)
            initial = sequence["state"][0:1].float().to(device)
            actions = sequence["joint_action"][None, None].float().to(device)
            rollout = model.rollout(initial, actions)
            prediction = rollout["predicted_states"][0, 0, 1:].cpu()
            target = sequence["next_state"].float()
            for group in ("robot_position", "ball_position", "ball_velocity"):
                indices = [i for f in model.schema.features if f.group == group for i in range(f.start, f.stop)]
                errors[group].append(float((prediction[:, indices] - target[:, indices]).square().mean().sqrt()))
            errors["cumulative_reward"].append(float((rollout["predicted_rewards"][0, 0].cpu().sum() - sequence["reward"].float().sum()).abs()))
            actual_done = (sequence["terminated"].bool() | sequence["truncated"].bool()).float()
            errors["termination_brier"].append(float((rollout["predicted_done_probabilities"][0, 0].cpu() - actual_done).square().mean()))
        results[str(horizon)] = {"count": len(errors["ball_position"]), **{key: float(np.mean(value)) for key, value in errors.items()}}
    return results


def save_plots(metrics, output):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    horizons = [int(key) for key, value in metrics["multi_step"].items() if value.get("count", 0)]
    if horizons:
        fig, axis = plt.subplots()
        for name in ("robot_position", "ball_position", "ball_velocity"):
            axis.plot(horizons, [metrics["multi_step"][str(h)][name] for h in horizons], marker="o", label=name)
        axis.set(xlabel="rollout horizon", ylabel="RMSE", title="World-model rollout error")
        axis.legend(); fig.tight_layout(); fig.savefig(output / "multi_step_error.png"); plt.close(fig)
    groups = metrics["one_step"]["groups"]
    fig, axis = plt.subplots()
    names = list(groups)
    axis.bar(names, [groups[name]["rmse"] for name in names])
    axis.set(ylabel="RMSE", title="One-step error by feature group")
    axis.tick_params(axis="x", rotation=25); fig.tight_layout(); fig.savefig(output / "one_step_feature_error.png"); plt.close(fig)
    reliability = [item for item in metrics["one_step"]["termination"]["reliability"] if item["count"]]
    if reliability:
        fig, axis = plt.subplots()
        axis.plot([item["predicted"] for item in reliability], [item["observed"] for item in reliability], marker="o")
        axis.plot([0, 1], [0, 1], linestyle="--", color="black")
        axis.set(xlabel="predicted probability", ylabel="observed frequency", title="Termination reliability")
        fig.tight_layout(); fig.savefig(output / "termination_reliability.png"); plt.close(fig)
    quantile_error = metrics["one_step"]["uncertainty"]["error_by_uncertainty_quintile"]
    fig, axis = plt.subplots()
    axis.plot(range(1, 6), quantile_error, marker="o")
    axis.set(xlabel="uncertainty quintile", ylabel="mean squared error", title="Uncertainty versus error")
    fig.tight_layout(); fig.savefig(output / "uncertainty_vs_error.png"); plt.close(fig)


def save_training_plot(checkpoint, output):
    history_path = Path(checkpoint).parent / "history.json"
    if not history_path.exists():
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    history = json.loads(history_path.read_text())
    fig, axis = plt.subplots()
    axis.plot([item["loss"] for item in history["train"]], label="train")
    axis.plot([item["loss"] for item in history["validation"]], label="validation")
    axis.set(xlabel="epoch", ylabel="loss", title="World-model training"); axis.legend()
    fig.tight_layout(); fig.savefig(output / "training_validation_loss.png"); plt.close(fig)


@torch.no_grad()
def save_trajectory_plots(model, dataset, device, output, horizon=20):
    available = dataset.sequences(horizon)
    if not available:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    episode_index, start = available[0]
    sequence = dataset.get_sequence(episode_index, start, horizon)
    initial = sequence["state"][0:1].float().to(device)
    rollout = model.rollout(initial, sequence["joint_action"][None, None].float().to(device))
    predicted = rollout["predicted_states"][0, 0, 1:].cpu()
    actual = sequence["next_state"].float()
    fig, axis = plt.subplots()
    ball_slice = model.schema.slice("ball.position")
    axis.plot(actual[:, ball_slice][:, 0], actual[:, ball_slice][:, 1], label="actual ball")
    axis.plot(predicted[:, ball_slice][:, 0], predicted[:, ball_slice][:, 1], linestyle="--", label="predicted ball")
    axis.set(xlabel="normalized field x", ylabel="normalized field y", title="Ball trajectory"); axis.legend()
    fig.tight_layout(); fig.savefig(output / "ball_trajectory.png"); plt.close(fig)
    fig, axis = plt.subplots()
    num_robots = model.action_adapter.num_robots
    for robot in range(num_robots):
        position = model.schema.slice(f"robot_{robot}.position")
        axis.plot(actual[:, position][:, 0], actual[:, position][:, 1], label=f"actual robot {robot}")
        axis.plot(predicted[:, position][:, 0], predicted[:, position][:, 1], linestyle="--", label=f"predicted robot {robot}")
    axis.set(xlabel="normalized field x", ylabel="normalized field y", title="Robot trajectories"); axis.legend()
    fig.tight_layout(); fig.savefig(output / "robot_trajectories.png"); plt.close(fig)
    fig, axis = plt.subplots()
    axis.plot(sequence["reward"].float(), label="actual")
    axis.plot(rollout["predicted_rewards"][0, 0].cpu(), label="predicted")
    axis.set(xlabel="macro step", ylabel="reward", title="Reward prediction"); axis.legend()
    fig.tight_layout(); fig.savefig(output / "reward_prediction.png"); plt.close(fig)


def main(args):
    device = args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    model, payload = load_checkpoint(args.checkpoint, device)
    model.eval()
    metadata = json.loads((Path(args.dataset) / "metadata.json").read_text())
    dataset_event_names = event_names_from_metadata(metadata)
    checkpoint_event_names = validate_event_names(payload.get("event_names", model.event_names))
    if checkpoint_event_names != dataset_event_names:
        raise ValueError(
            "Checkpoint and dataset event schemas differ: "
            f"checkpoint={checkpoint_event_names}, dataset={dataset_event_names}"
        )
    dataset = WorldModelDataset(args.dataset, args.split)
    horizons = payload["training_config"].get("evaluation", {}).get("horizons", [1, 3, 5, 10, 20])
    metrics = {
        "one_step": one_step(model, dataset, device, args.batch_size, dataset_event_names),
        "multi_step": multi_step(model, dataset, device, horizons),
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    save_plots(metrics, output)
    save_training_plot(args.checkpoint, output)
    save_trajectory_plots(model, dataset, device, output)
    print(json.dumps(metrics, indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/world_model_as2/best.pt")
    parser.add_argument("--dataset", default="data/world_model_as2")
    parser.add_argument("--split", default="test", choices=("train", "validation", "test"))
    parser.add_argument("--output", default="outputs/world_model_as2_evaluation")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2048)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
