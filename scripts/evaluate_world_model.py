"""Evaluate open-loop endpoint errors and inference time on held-out episodes."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from quadruped.mpc.terminal_value import compute_discounted_returns, load_value_checkpoint
from quadruped.world_model.dataset import WorldModelDataset
from quadruped.world_model.trainer import load_checkpoint
from quadruped.world_model.schema import StateSchema


def align_dataset_schema(dataset, source_payload, target, allow_missing_shoot_timers=False):
    """Map legacy fields by name; never infer unrecorded active option timers."""
    source = StateSchema.from_dict(source_payload)
    if source.to_dict() == target.to_dict():
        return []
    source_features = {f.name: f for f in source.features}
    target_names = {f.name for f in target.features}
    missing = [f.name for f in target.features if f.name not in source_features]
    extra = sorted(set(source_features) - target_names)
    changed = [f.name for f in target.features if f.name in source_features and
               any(getattr(f, key) != getattr(source_features[f.name], key)
                   for key in ("size", "kind", "unit", "dynamic", "group"))]
    detail = (f"Dataset state dimension {source.state_dim}, checkpoint {target.state_dim}; "
              f"missing fields: {missing}; extra fields: {extra}; changed fields: {changed}.")
    supported = all(name.startswith("robot_") and name.endswith(".shoot_option_remaining")
                    and target.feature(name).size == 1 for name in missing)
    if extra or changed or source.version != target.version or not supported:
        raise ValueError(detail + " Use a dataset matching this checkpoint.")
    if missing and not allow_missing_shoot_timers:
        raise ValueError(detail + " Use matching data, or pass --allow-missing-shoot-timers "
                         "to explicitly assume zero (inactive) timers for legacy data.")
    for episode in dataset.episodes:
        for key in ("state", "next_state"):
            original = episode[key]
            if original.shape[-1] != source.state_dim:
                raise ValueError(f"Dataset {key} width differs from its metadata")
            mapped = np.zeros((*original.shape[:-1], target.state_dim), dtype=original.dtype)
            for feature in target.features:
                if feature.name in source_features:
                    old = source_features[feature.name]
                    mapped[..., feature.start:feature.stop] = original[..., old.start:old.stop]
            episode[key] = mapped
    return missing


def position_errors(schema, predicted, actual, group):
    """Return Euclidean XYZ distances in metres, one per sample/object."""
    scale = torch.ones_like(actual[..., :3])
    scale[..., :2] = actual[..., schema.slice("field.geometry")][..., :2]
    features = [f for f in schema.features if f.group == group]
    if not features:
        raise ValueError(f"Schema has no {group} features")
    return torch.stack([
        ((predicted[..., f.start:f.stop] - actual[..., f.start:f.stop]) * scale).norm(dim=-1)
        for f in features
    ], dim=-1)


def continuation_targets(episode, gamma):
    """V(s_t) targets, including zero at a true terminal endpoint.

    Truncated/incomplete episodes have unknown continuation and are excluded.
    """
    count = len(episode["reward"])
    if not bool(episode["terminated"][-1]):
        return np.full(count + 1, np.nan, dtype=np.float32)
    returns = compute_discounted_returns(
        episode["reward"], episode["terminated"], episode["truncated"], gamma
    )
    return np.append(returns, np.float32(0))


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def endpoint_return_targets(episode, value_model, gamma, device, bootstrap_on_timeout=True):
    """Return G_t through G_T, with a fixed real-endpoint timeout bootstrap."""
    terminal, timeout = bool(episode["terminated"][-1]), bool(episode["truncated"][-1])
    if not (terminal or timeout):
        return np.full(len(episode["reward"]) + 1, np.nan, dtype=np.float32)
    bootstrap = 0.0
    if timeout and not terminal and bootstrap_on_timeout:
        bootstrap = float(value_model.predict(torch.as_tensor(
            episode["next_state"][-1:], device=device).float()).reshape(-1)[0])
    returns = compute_discounted_returns(
        episode["reward"], episode["terminated"], episode["truncated"], gamma,
        bootstrap_value=bootstrap, bootstrap_on_truncation=bootstrap_on_timeout)
    return np.append(returns, np.float32(bootstrap))


def uncertainty_metrics(error, variance):
    """Diagonal Gaussian diagnostics; correlations are undefined for constants."""
    error, variance = error.double().flatten(), variance.double().flatten()
    valid = torch.isfinite(error) & torch.isfinite(variance) & (variance >= 0)
    error, variance = error[valid], variance[valid]
    if not error.numel():
        return {"samples": 0, "mean_std": None, "error_correlation": None,
                "coverage_95": None, "gaussian_nll": None}
    squared = error.square()
    correlation = None
    if error.numel() > 1 and variance.std() > 0 and squared.std() > 0:
        correlation = float(torch.corrcoef(torch.stack((variance, squared)))[0, 1])
    safe = variance.clamp(min=1e-12)
    return {"samples": error.numel(), "mean_std": float(variance.sqrt().mean()),
            "error_correlation": correlation,
            "coverage_95": float((error.abs() <= 1.96 * variance.sqrt()).double().mean()),
            "gaussian_nll": float((.5 * (np.log(2 * np.pi) + safe.log() + squared / safe)).mean())}


@torch.inference_mode()
def ball_uncertainty(model, initial, actions, actual, particles=128):
    """Propagate particles through the ensemble's moment-matched transition law."""
    if particles < 2:
        raise ValueError("uncertainty particles must be at least 2")
    batch, width = initial.shape
    current = initial[:, None].expand(-1, particles, -1).reshape(-1, width).clone()
    for step in range(actions.shape[2]):
        action = actions[:, 0, step, None].expand(-1, particles, -1).reshape(batch * particles, -1)
        current, _, _, _, _ = model.predict_next(current, action, deterministic=False)
    scale = torch.ones_like(actual[:, :3])
    scale[:, :2] = actual[:, model.schema.slice("field.geometry")][:, :2]
    endpoints = current.reshape(batch, particles, width)[..., model.schema.slice("ball.position")]
    error = (endpoints.mean(1) - actual[:, model.schema.slice("ball.position")]) * scale
    variance = endpoints.var(1, unbiased=True) * scale.square()
    return error.cpu(), variance.cpu()


@torch.inference_mode()
def training_value_metrics(value_model, dataset, gamma, device, batch_size=128,
                           bootstrap_on_timeout=True):
    """Match training's _observe_value_quality and the plotted MSE/variance.

    A loaded value checkpoint is treated as ready for timeout bootstrapping.
    Episode EMA follows manifest order; no extra plot-window smoothing is used.
    """
    quality = None
    samples = 0
    for episode in dataset.episodes:
        terminated = bool(episode["terminated"][-1])
        timeout = bool(episode["truncated"][-1])
        if not (terminated or timeout):
            continue
        bootstrap = 0.0
        if timeout and not terminated and bootstrap_on_timeout:
            endpoint = torch.as_tensor(episode["next_state"][-1:], device=device).float()
            bootstrap = float(value_model.predict(endpoint).reshape(-1)[0])
        returns = compute_discounted_returns(
            episode["reward"], episode["terminated"], episode["truncated"], gamma,
            bootstrap_value=bootstrap, bootstrap_on_truncation=bootstrap_on_timeout)
        target = torch.as_tensor(returns, device=device).float()
        prediction = torch.cat([
            value_model.predict(torch.as_tensor(episode["state"][start:start + batch_size],
                                                device=device).float()).reshape(-1)
            for start in range(0, len(returns), batch_size)])
        measurements = {"mse": float((prediction - target).square().mean()),
                        "mean": float(target.mean()), "second": float(target.square().mean())}
        if quality is None:
            quality = measurements.copy()
        for key, value in measurements.items():
            quality[key] = .95 * quality[key] + .05 * value
        samples += len(returns)
    if quality is None:
        return {"terminal_value_samples": 0, "terminal_value_mse": None,
                "terminal_value_variance": None, "terminal_value_normalized_mse": None}
    variance = max(quality["second"] - quality["mean"] ** 2, 1e-6)
    result = {"terminal_value_samples": samples, "terminal_value_mse": quality["mse"],
            "terminal_value_variance": variance,
            "terminal_value_normalized_mse": quality["mse"] / variance}
    return result


@torch.inference_mode()
def benchmark_mpc(model, dataset, horizons, config, *, device, samples=3,
                  warmup=1, repeats=3, seed=42, value_model=None):
    """Time complete cold-start CEM planning calls, one state per inference."""
    from quadruped.mpc.hybrid_cem import HybridCEMMPC
    from quadruped.mpc.objective import MPCObjective

    device = torch.device(device)
    origins = dataset.sequences(max(horizons))
    selected = np.random.default_rng(seed).permutation(len(origins))[:samples]
    states = [dataset.get_sequence(*origins[i], 1)["state"].float().to(device)
              for i in selected]
    if not states:
        raise ValueError("No valid states for MPC timing")
    results, configs = {}, {}
    for horizon in horizons:
        cfg = replace(config, horizon=horizon, warm_start=False, seed=seed).validate()
        objective = MPCObjective(model.schema, model.action_adapter, model.event_names, cfg,
                                 terminal_value=value_model.predict_with_uncertainty if value_model is not None else None)
        planner = HybridCEMMPC(model, objective=objective, config=cfg)
        for _ in range(warmup):
            planner.plan(states[0])
        timings = []
        for state in states:
            for _ in range(repeats):
                synchronize(device)
                started = time.perf_counter()
                planner.plan(state)
                synchronize(device)
                timings.append((time.perf_counter() - started) * 1000)
        results[horizon] = {
            "mpc_ms_per_inference": float(np.mean(timings)),
            "mpc_p95_ms_per_inference": float(np.percentile(timings, 95)),
            "mpc_timed_inferences": len(timings),
        }
        configs[horizon] = cfg.to_dict()
    return results, configs


@torch.inference_mode()
def evaluate(model, dataset, horizons, *, device, batch_size=128, max_samples=2048,
             seed=42, warmup=3, repeats=5, value_model=None, gamma=0.99,
             bootstrap_on_timeout=True, uncertainty_particles=128):
    device = torch.device(device)
    model.eval()
    if value_model is not None:
        value_model.eval()
    # Identical origins across horizons make both errors and timing comparable.
    origins = dataset.sequences(max(horizons))
    if not origins:
        raise ValueError(f"No uninterrupted sequences of length {max(horizons)} in split")
    rng = np.random.default_rng(seed)
    chosen = rng.permutation(len(origins))[:max_samples]
    origins = [origins[i] for i in chosen]
    targets = ([endpoint_return_targets(ep, value_model, gamma, device, bootstrap_on_timeout)
                for ep in dataset.episodes] if value_model is not None else [])
    rows = []
    for horizon in horizons:
        ball_errors, robot_errors = [], []
        uncertainty_batches = []
        value_errors, value_targets, value_deltas, real_values = [], [], [], []
        rollout_ms, value_ms, batch_count = 0.0, 0.0, 0
        for offset in range(0, len(origins), batch_size):
            selected = origins[offset:offset + batch_size]
            sequences = [dataset.get_sequence(ep, start, horizon) for ep, start in selected]
            initial = torch.stack([s["state"][0] for s in sequences]).float().to(device)
            actions = torch.stack([s["joint_action"] for s in sequences]).float().to(device)[:, None]
            actual = torch.stack([s["next_state"][-1] for s in sequences]).float().to(device)

            def predict():
                return model.rollout(initial, actions, deterministic=True, stop_on_done=False)["predicted_states"][:, 0, -1]

            for _ in range(warmup):
                predicted = predict()
                if value_model is not None:
                    value_model.predict(predicted)
            synchronize(device)
            start_time = time.perf_counter()
            for _ in range(repeats):
                predicted = predict()
            synchronize(device)
            rollout_ms += (time.perf_counter() - start_time) * 1000 / repeats
            batch_count += 1
            if value_model is not None:
                synchronize(device)
                start_time = time.perf_counter()
                for _ in range(repeats):
                    values = value_model.predict(predicted).reshape(-1)
                synchronize(device)
                value_ms += (time.perf_counter() - start_time) * 1000 / repeats
                # Score rollout-induced value distortion separately from critic error.
                real_value = value_model.predict(actual).reshape(-1)
                value_deltas.append((values - real_value).cpu())
                real_values.append(real_value.cpu())
                target = torch.tensor([targets[ep][start + horizon] for ep, start in selected],
                                      device=device, dtype=values.dtype)
                valid = torch.isfinite(target)
                value_errors.append((values[valid] - target[valid]).cpu())
                value_targets.append(target[valid].cpu())
            ball_errors.append(position_errors(model.schema, predicted, actual, "ball_position").cpu())
            robot_errors.append(position_errors(model.schema, predicted, actual, "robot_position").cpu())
            if hasattr(model, "predict_next"):
                # Bound particle memory independently of the inference batch size.
                with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
                    torch.manual_seed(seed + offset)
                    for start in range(0, len(initial), 8):
                        uncertainty_batches.append(ball_uncertainty(
                            model, initial[start:start + 8], actions[start:start + 8],
                            actual[start:start + 8], uncertainty_particles))
        row = {"horizon": horizon, "samples": len(origins)}
        for name, parts in (("ball", ball_errors), ("robot", robot_errors)):
            distances = torch.cat(parts)
            row[f"{name}_mean_distance_m"] = float(distances.mean())
            row[f"{name}_rmse_m"] = float(distances.square().mean().sqrt())
        row.update(
            rollout_ms_per_batch=rollout_ms / batch_count,
            rollout_ms_per_sample=rollout_ms / len(origins),
            value_ms_per_batch=value_ms / batch_count if value_model is not None else None,
            total_ms_per_sample=(rollout_ms + value_ms) / len(origins),
        )
        errors = torch.cat(value_errors).double() if value_errors else torch.empty(0)
        target = torch.cat(value_targets).double() if value_targets else torch.empty(0)
        mse = float(errors.square().mean()) if errors.numel() else None
        variance = max(float(target.var(unbiased=False)), 1e-6) if target.numel() else None
        row.update(terminal_value_samples=errors.numel(), terminal_value_mse=mse,
                   terminal_value_variance=variance,
                   terminal_value_normalized_mse=mse / variance if mse is not None else None)
        delta = torch.cat(value_deltas).double() if value_deltas else torch.empty(0)
        real_value = torch.cat(real_values).double() if real_values else torch.empty(0)
        row.update(
            value_rollout_samples=delta.numel(),
            value_rollout_rmse=float(delta.square().mean().sqrt()) if delta.numel() else None,
            value_rollout_mae=float(delta.abs().mean()) if delta.numel() else None,
            real_endpoint_value_std=float(real_value.std(unbiased=False)) if real_value.numel() else None,
        )
        if uncertainty_batches:
            error = torch.cat([batch[0] for batch in uncertainty_batches])
            variance = torch.cat([batch[1] for batch in uncertainty_batches])
            row.update({f"ball_accumulated_uncertainty_{key}": value
                        for key, value in uncertainty_metrics(error, variance).items()})
        rows.append(row)
    # One reference scale for every horizon: observed value spread at shortest H.
    reference = min(rows, key=lambda row: row["horizon"])
    reference_std = reference["real_endpoint_value_std"]
    for row in rows:
        row["value_rollout_reference_horizon"] = reference["horizon"]
        row["value_rollout_reference_std"] = reference_std
        row["value_rollout_nrmse"] = (
            row["value_rollout_rmse"] / reference_std
            if reference_std is not None and reference_std > 1e-6 else None)
    return rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--value-checkpoint", help="Separate terminal V(s) checkpoint; otherwise value errors are null")
    parser.add_argument("--allow-missing-shoot-timers", action="store_true",
                        help="Map legacy states by name and assume missing shooting-option timers are zero")
    parser.add_argument("--split", choices=("test", "validation", "train"), default="test")
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 2, 4, 10])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-samples", type=int, default=2048)
    parser.add_argument("--uncertainty-particles", type=int, default=128,
                        help="Stochastic trajectories per origin for accumulated ball uncertainty")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="outputs/world_model_evaluation")
    parser.add_argument("--mpc-config", default="configs/mpc.yaml",
                        help="Search configuration for full MPC ms/inference benchmark")
    parser.add_argument("--mpc-profile", default=None)
    parser.add_argument("--mpc-timing-samples", type=int, default=3,
                        help="Number of held-out states for planning latency")
    parser.add_argument("--mpc-timing-repeats", type=int, default=3)
    parser.add_argument("--mpc-timing-warmup", type=int, default=1)
    parser.add_argument("--skip-mpc-timing", action="store_true")
    parser.add_argument("--no-terminal-bootstrap-on-timeout", action="store_true",
                        help="Match training runs that disabled timeout value bootstrapping")
    args = parser.parse_args(argv)
    if args.uncertainty_particles < 2:
        parser.error("--uncertainty-particles must be at least 2")
    if min(args.horizons + [args.batch_size, args.max_samples, args.repeats]) < 1 or args.warmup < 0:
        parser.error("horizons, batch-size, max-samples and repeats must be positive; warmup must be nonnegative")
    if min(args.mpc_timing_samples, args.mpc_timing_repeats) < 1 or args.mpc_timing_warmup < 0:
        parser.error("MPC timing samples/repeats must be positive and warmup nonnegative")
    args.horizons = sorted(set(args.horizons))
    return args


def main(args):
    model, _ = load_checkpoint(args.checkpoint, args.device)
    value_model, gamma = None, 0.99
    if args.value_checkpoint:
        value_model, payload = load_value_checkpoint(args.value_checkpoint, args.device)
        if value_model.schema.to_dict() != model.schema.to_dict():
            raise ValueError("Dynamics and value checkpoint schemas do not match")
        gamma = float(payload["gamma"])
    dataset = WorldModelDataset(args.dataset, args.split)
    metadata_path = Path(args.dataset) / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    filled_fields = align_dataset_schema(dataset, metadata["state_schema"], model.schema,
                                         args.allow_missing_shoot_timers)
    if filled_fields:
        print("Legacy dataset: assuming zero (inactive) timers for " + ", ".join(filled_fields)
              + ". This does not evaluate active shooting-option trajectories.")
    if metadata["action_schema"] != model.action_adapter.to_dict():
        raise ValueError("Dataset and checkpoint action schemas do not match")
    rows = evaluate(model, dataset, args.horizons, device=args.device,
                    batch_size=args.batch_size, max_samples=args.max_samples,
                    seed=args.seed, warmup=args.warmup, repeats=args.repeats,
                    value_model=value_model, gamma=gamma,
                    bootstrap_on_timeout=not args.no_terminal_bootstrap_on_timeout,
                    uncertainty_particles=args.uncertainty_particles)
    value_status = "available"
    if value_model is None:
        value_status = "No --value-checkpoint supplied"
    elif not any(row["terminal_value_samples"] for row in rows):
        value_status = "No complete terminated or truncated episodes for return targets."
    if value_status != "available":
        print("Value normalized MSE N/A: " + value_status)
    planning_configs = {}
    if not args.skip_mpc_timing:
        from quadruped.mpc.config import load_mpc_config
        config, _ = load_mpc_config(args.mpc_config, args.mpc_profile)
        config.shooting_options = any(f.name.endswith(".shoot_option_remaining") for f in model.schema.features)
        print(f"Timing full MPC: {config.num_candidates} candidates, {config.num_iterations} CEM iterations, "
              f"candidate batches of {config.candidate_batch_size or config.num_candidates}; "
              "one state per inference, cold start, jointly optimizing all robots.")
        timings, planning_configs = benchmark_mpc(
            model, dataset, args.horizons, config, device=args.device,
            samples=args.mpc_timing_samples, repeats=args.mpc_timing_repeats,
            warmup=args.mpc_timing_warmup, seed=args.seed, value_model=value_model)
        for row in rows:
            row.update(timings[row["horizon"]])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    report = {"arguments": vars(args), "gamma": gamma if value_model is not None else None,
              "schema_adaptation": {"zero_filled_fields": filled_fields},
              "terminal_value_status": value_status, "mpc_configs": planning_configs, "metrics": rows}
    report["terminal_value_definition"] = {
        "primary_metric": "sqrt(mean((V(predicted_state_t+H) - V(real_state_t+H))^2))",
        "primary_interpretation": "rollout-induced value distortion, not accuracy against realized returns",
        "normalization": "divide rollout RMSE by real-endpoint value std at shortest requested horizon, fixed across horizons; null if <=1e-6",
        "metric": "mean((V(predicted_state_t+H) - G_t+H)^2) / max(var(G_t+H), 1e-6)",
        "states": "predicted H-step endpoints at shared sampled origins; targets start at corresponding real endpoints",
        "aggregation": "pooled samples per horizon, no temporal EMA",
        "bootstrap_on_timeout": not args.no_terminal_bootstrap_on_timeout,
        "checkpoint_assumed_ready": True, "additional_plot_window_smoothing": False}
    report["uncertainty_definition"] = {
        "position": "ball endpoint sample variance across stochastic trajectories propagated for H steps",
        "position_units": "coordinate errors/std in metres; variances in square metres",
        "sampling": "moment-matched diagonal Gaussian with aleatoric + epistemic variance at each sampled state; no persistent ensemble member identity",
        "particles": args.uncertainty_particles,
        "coverage": "coordinate-wise Gaussian approximation about particle endpoint mean at +/-1.96 std",
        "correlation": "Pearson correlation of predicted variance with squared error; null when constant"}
    (output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with (output / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Horizon  Samples  Ball RMSE(m)  Robot RMSE(m)  Value rollout RMSE  Value rollout NRMSE  Ball accumulated SD(m)  MPC ms/inference")
    for row in rows:
        value = row["value_rollout_nrmse"]
        value_text = f"{value:.5f}" if value is not None else "N/A"
        mse = row["value_rollout_rmse"]
        mse_text = f"{mse:.5f}" if mse is not None else "N/A"
        latency = row.get("mpc_ms_per_inference")
        latency_text = f"{latency:.3f}" if latency is not None else "N/A"
        uncertainty = row.get("ball_accumulated_uncertainty_mean_std")
        uncertainty_text = f"{uncertainty:.5f}" if uncertainty is not None else "N/A"
        print(f"{row['horizon']:7d}  {row['samples']:7d}  {row['ball_rmse_m']:12.5f}  "
              f"{row['robot_rmse_m']:13.5f}  {mse_text:>18}  {value_text:>19}  {uncertainty_text:>22}  {latency_text:>16}")
    print("Value rollout RMSE compares V(predicted endpoint) with V(real endpoint); "
          "NRMSE uses the same reference scale for every horizon. Return-target MSE remains in JSON/CSV.")
    print("Ball accumulated SD = mean per-coordinate endpoint standard deviation across stochastic H-step trajectories (metres).")
    print(f"Reports: {output / 'metrics.json'} and {output / 'metrics.csv'}")
    for row in rows:
        if "ball_accumulated_uncertainty_coverage_95" in row:
            print(f"H={row['horizon']} uncertainty 95% coverage: "
                  f"ball={row['ball_accumulated_uncertainty_coverage_95']:.3f}; "
                  f"variance/error correlation: ball={row['ball_accumulated_uncertainty_error_correlation']}")


if __name__ == "__main__":
    main(parse_args())
