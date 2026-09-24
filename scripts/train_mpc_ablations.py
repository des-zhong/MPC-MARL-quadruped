"""Train matched MPC ablations sequentially; use --dry-run to inspect commands."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_mpc_efficiency_ablation import command

CONDITIONS = {
    "no_uncertainty": ("disabled", "enabled"),
    "no_terminal_value": ("enabled", "disabled"),
    "full": ("enabled", "enabled"),
}


def build_jobs(args):
    jobs = []
    for seed in args.seeds:
        for condition in args.conditions:
            uncertainty, terminal = CONDITIONS[condition]
            output = args.output_root / condition / f"seed_{seed}"
            argv = command(
                "mpc", seed, args.iterations, args.num_envs, f"cuda:{args.cuda}",
                shoot_policy_dir=args.shoot_policy_dir,
                mpc_performance=args.mpc_performance,
            )
            argv += [
                "--world-model-checkpoint", args.world_model_checkpoint,
                "--mpc-uncertainty", uncertainty,
                "--mpc-terminal-value", terminal,
                "--checkpoint-dir", str(output),
                "--project", f"{args.project_prefix}_{condition}",
                "--save-video-interval", str(args.save_video_interval),
            ]
            jobs.append(dict(condition=condition, seed=seed, argv=argv,
                             checkpoint_dir=str(output)))
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS,
                        default=["no_uncertainty", "no_terminal_value"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--cuda", type=int,
                        default=int(os.environ.get("DRIBBLEBOT_CUDA_INDEX", 7)))
    parser.add_argument("--iterations", type=int, default=6000)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--world-model-checkpoint", default="checkpoints/offline_teacher/best.pt")
    parser.add_argument("--shoot-policy-dir", default="checkpoints/reproduction/shoot")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/mpc_ablations"))
    parser.add_argument("--project-prefix", default="as2_mpc_ablation")
    parser.add_argument("--save-video-interval", type=int, default=0)
    parser.add_argument("--mpc-performance", action="store_true",
                        help="Apply the extra-compute MPC preset to every condition.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.cuda < 0 or args.iterations < 1 or args.num_envs < 1:
        parser.error("GPU index must be nonnegative; iterations and num-envs must be positive")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.conditions)) != len(args.conditions):
        parser.error("seeds and conditions must be unique")
    if not args.output_root.is_absolute():
        args.output_root = ROOT / args.output_root
    jobs = build_jobs(args)
    for job in jobs:
        print(shlex.join(job["argv"]), flush=True)
    if args.dry_run:
        return
    # Refuse to mix new training with existing runs. Choose another output root
    # for a repeat experiment or a different search budget.
    for job in jobs:
        output = Path(job["checkpoint_dir"])
        if output.exists() and any(output.iterdir()):
            parser.error(f"run directory is not empty: {output}; choose a new --output-root")
    args.output_root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/dribblebot_torch_extensions")
    for job in jobs:
        output = Path(job["checkpoint_dir"])
        output.mkdir(parents=True, exist_ok=True)
        (output / "launch.json").write_text(json.dumps(job, indent=2) + "\n")
        print(f"Training {job['condition']}, seed {job['seed']}: {output}", flush=True)
        subprocess.run(job["argv"], cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
