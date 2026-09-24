"""Compare training reward spikes averaged over (start, end] windows.

goal_rate.csv reports estimated goals per 100 matches:
100 * mean(rew_high_level_goal / configured_goal_reward).
Each logged reset-batch mean has equal weight; historical logs cannot recover
the exact match-weighted rate because reset-batch sizes were not saved.
Empty or incomplete windows remain blank. Zero means observed zero goals.
"""
import argparse
import csv
import json
import math
from pathlib import Path


METRICS = {
    "goal_rate": "high_level_goal",
    "termination_rate": "high_level_accidental_termination",
}


def local_history(path):
    """Scan a temporary snapshot so the original run file is never opened writable."""
    import shutil
    import tempfile
    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal.datastore import DataStore
    from wandb.sdk.lib import proto_util

    with tempfile.TemporaryDirectory() as directory:
        snapshot = Path(directory) / "history.wandb"
        shutil.copyfile(path, snapshot)
        store = DataStore()
        store.open_for_scan(str(snapshot))
        try:
            while True:
                try:
                    data = store.scan_data()
                except AssertionError:
                    if store.in_last_block():
                        print(f"Skipping unfinished final record in {path}")
                        break
                    raise
                if data is None:
                    break
                record = wandb_internal_pb2.Record()
                record.ParseFromString(data)
                if record.WhichOneof("record_type") == "history":
                    yield proto_util.dict_from_proto_list(record.history.item)
        finally:
            store.close()


def source_data(source):
    path = Path(source).expanduser()
    if path.is_dir():
        import yaml
        files = list(path.glob("run-*.wandb"))
        if len(files) != 1:
            raise ValueError(f"Expected one run-*.wandb file in {path}")
        with (path / "files/config.yaml").open() as file:
            config = yaml.safe_load(file)
        return local_history(files[0]), config
    if source.startswith((".", "/", "wandb/")):
        raise FileNotFoundError(source)
    import wandb
    run = wandb.Api().run(source)
    return run.scan_history(), dict(run.config)


def unwrap(value):
    return value.get("value", value) if isinstance(value, dict) else value


def read_run(source, x_axis):
    history, config = source_data(source)
    cfg = unwrap(config["Cfg"])
    rewards = unwrap(cfg["rewards"])
    scales = {}
    for metric, term in METRICS.items():
        if term not in rewards.get("unscaled_reward_names", []):
            raise ValueError(f"{source}: {term} is not an unscaled event reward")
        scale = float(unwrap(cfg.get("reward_scales", rewards))[term])
        if not math.isfinite(scale) or scale == 0:
            raise ValueError(f"{source}: invalid reward scale for {term}")
        scales[metric] = scale
    merged = {}
    keys = ("iterations", "timesteps", *(f"rew_{term}" for term in METRICS.values()))
    for record in history:
        if "_step" in record:
            merged.setdefault(record["_step"], {}).update(
                {key: record[key] for key in keys if key in record}
            )
    axis_key = "iterations" if x_axis == "epochs" else "timesteps"
    rows = []
    for record in merged.values():
        if axis_key not in record:
            continue
        x = float(record[axis_key])
        if not math.isfinite(x) or x < 0 or not x.is_integer():
            raise ValueError(f"Invalid {axis_key}: {x}")
        row = {"x": int(x)}
        for metric, term in METRICS.items():
            value = record.get(f"rew_{term}")
            if value is not None and math.isfinite(float(value)):
                rate = 100 * float(value) / scales[metric]
                if not -1e-4 <= rate <= 100.0001:
                    raise ValueError(f"{source}: {term} implies invalid rate {rate}")
                row[metric] = min(100., max(0., rate))
        rows.append(row)
    if not rows:
        raise ValueError(f"No {axis_key} history in {source}")
    return sorted(rows, key=lambda row: row["x"])


def aggregate(rows, period, metric):
    """Only report full windows; epoch zero belongs to the first window."""
    last_x = max(row["x"] for row in rows)
    groups = {}
    for row in rows:
        end = max(1, (row["x"] + period - 1) // period) * period
        if end <= last_x and metric in row:
            groups.setdefault(end, []).append(row[metric])
    return {end: math.fsum(values) / len(values) for end, values in groups.items()}


def record(ppo_run, mpc_run, output_dir, period=100, x_axis="epochs"):
    if period <= 0:
        raise ValueError("--period must be positive")
    histories = {method: read_run(source, x_axis)
                 for method, source in (("ppo", ppo_run), ("mpc", mpc_run))}
    last = max(row["x"] for rows in histories.values() for row in rows)
    ends = list(range(period, last + 1, period))
    if not ends:
        raise ValueError(f"No complete {period}-{x_axis} window; last {x_axis}={last}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for metric in METRICS:
        with (output / f"{metric}.csv").open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["method", *ends])
            for method in ("ppo", "mpc"):
                values = aggregate(histories[method], period, metric)
                writer.writerow([method, *(format(values[end], ".10g")
                                            if end in values else "" for end in ends)])
    metadata = {
        "x_axis": x_axis, "period": period, "windows": "(end-period, end]",
        "units": "estimated events per 100 matches",
        "formula": "100 * mean(logged episode reward / configured event reward)",
        "weighting": "Equal weight per logged reset-batch mean, not per match",
        "blank": "No observations or incomplete window; do not replace with zero",
        "sources": {"ppo": ppo_run, "mpc": mpc_run},
    }
    (output / "training_rates_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote goal_rate.csv and termination_rate.csv in {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ppo-run", required=True, help="Local run directory or entity/project/id")
    parser.add_argument("--mpc-run", required=True, help="Local run directory or entity/project/id")
    parser.add_argument("--output-dir", default="outputs/training_rates")
    parser.add_argument("--x-axis", choices=("epochs", "timesteps"), default="epochs",
                        help="epochs uses the logged PPO iteration; timesteps uses agent transitions")
    parser.add_argument("--period", type=int, default=100, help="Window width in chosen x-axis units")
    args = parser.parse_args()
    record(args.ppo_run, args.mpc_run, args.output_dir, args.period, args.x_axis)


if __name__ == "__main__":
    main()
