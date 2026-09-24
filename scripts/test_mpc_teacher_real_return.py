"""Paired, first-action MPC intervention with independent simulator replays.

No physics clone is assumed. Student, teacher, and student-repeat workers each
reconstruct the same seeded prefix in a fresh process. The coordinator checks
their exposed simulator/controller states and the student repeat before making
any return comparison. Run --help without loading Isaac Gym or a GPU.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def compare_snapshots(left, right, tolerance, force_tolerance=None):
    """Reject missing tensors, nonfinite values, shape changes, and drift."""
    import numpy as np

    failures = {}
    maximum = 0.0
    with np.load(left) as a, np.load(right) as b:
        for key in sorted(set(a.files) | set(b.files)):
            if key not in a or key not in b or a[key].shape != b[key].shape:
                failures[key] = "missing or shape mismatch"
                continue
            x, y = a[key], b[key]
            if not (np.isfinite(x).all() and np.isfinite(y).all()):
                failures[key] = "nonfinite"
                continue
            error = float(np.max(np.abs(x.astype(float) - y.astype(float)))) if x.size else 0.0
            maximum = max(maximum, error)
            # GPU contact-force reductions can differ by a few float32 ULPs
            # even when joint/root states and the complete sham trajectory agree.
            limit = force_tolerance if force_tolerance is not None and "contact_forces" in key else tolerance
            if error > limit:
                failures[key] = error
    return {"passed": not failures, "max_abs_difference": maximum, "failures": failures}


def real_returns(rewards, dones, gamma, short_horizon):
    """Include terminal rewards; never leak rewards from auto-reset episodes."""
    import numpy as np

    rewards, dones = np.asarray(rewards), np.asarray(dones, dtype=bool)
    if rewards.ndim != 2 or rewards.shape != dones.shape or len(rewards) == 0:
        raise ValueError("Expected nonempty [step, match] reward/done arrays")
    active = np.ones_like(dones)
    active[1:] = np.logical_and.accumulate(~dones[:-1], axis=0)
    discounted = rewards * active * np.power(gamma, np.arange(len(rewards)))[:, None]
    return {"return": discounted.sum(0), "short_return": discounted[:short_horizon].sum(0),
            "steps": active.sum(0), "censored": ~dones.any(0)}


def isolated_random_call(callback, seed, devices=()):
    """Reset draws must not shift noise for still-active neighboring matches."""
    import random
    import numpy as np
    import torch

    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=list(devices)):
            random.seed(seed)
            np.random.seed(seed % (2 ** 32))
            torch.random.default_generator.manual_seed(seed)
            for device in devices:
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(seed)
            return callback()
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def unchanged_match_check(left, right, matches, tolerance):
    """Other interventions must not perturb an unchanged match before it ends."""
    import numpy as np

    failures = {}
    with np.load(left) as a, np.load(right) as b:
        steps, batch = a["done"].shape
        active = np.ones_like(a["done"], dtype=bool)
        active[1:] = np.logical_and.accumulate(~a["done"][:-1], axis=0)
        selected = np.zeros(batch, dtype=bool)
        selected[matches] = True
        mask = active & selected[None]
        for key in ("state", "reward", "done", "actions"):
            x = a[key].reshape(steps, batch, -1)[mask].astype(float)
            y = b[key].reshape(steps, batch, -1)[mask].astype(float)
            error = float(np.max(np.abs(x - y))) if x.size else 0.0
            if not np.isfinite(error) or error > tolerance:
                failures[key] = error if np.isfinite(error) else "nonfinite"
    return {"passed": not failures, "match_count": len(matches), "failures": failures}


def summarize(rows, bootstrap_seed=123):
    """Bootstrap seed batches, preserving within-batch dependence."""
    import numpy as np

    if not rows:
        return {"count": 0, "mean_real_advantage": None, "seed_bootstrap_95ci": None}
    delta = np.array([r["real_advantage"] for r in rows])
    predicted = np.array([r["predicted_advantage"] if r.get("predicted_advantage_finite", True) else np.nan for r in rows])
    seeds = sorted({r["seed"] for r in rows})
    interval = None
    if len(seeds) >= 2:
        groups = [np.array([r["real_advantage"] for r in rows if r["seed"] == s]) for s in seeds]
        rng = np.random.default_rng(bootstrap_seed)
        estimates = [float(np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))]).mean())
                     for _ in range(2000)]
        interval = np.percentile(estimates, [2.5, 97.5]).tolist()
    correlation = None
    finite = np.isfinite(predicted)
    if finite.sum() > 2 and np.std(delta[finite]) > 0 and np.std(predicted[finite]) > 0:
        correlation = float(np.corrcoef(predicted[finite], delta[finite])[0, 1])
    return {
        "count": len(rows), "seed_batches": len(seeds),
        "mean_real_advantage": float(delta.mean()),
        "median_real_advantage": float(np.median(delta)),
        "seed_bootstrap_95ci": interval,
        "positive_real_advantage_fraction": float((delta > 1e-6).mean()),
        "negative_real_advantage_fraction": float((delta < -1e-6).mean()),
        "mean_predicted_advantage": float(predicted[finite].mean()) if finite.any() else None,
        "predicted_real_advantage_correlation": correlation,
        "teacher_goal_rate": float(np.mean([r["teacher_goal"] for r in rows])),
        "student_goal_rate": float(np.mean([r["student_goal"] for r in rows])),
        "teacher_censored_fraction": float(np.mean([r["teacher_censored"] for r in rows])),
        "student_censored_fraction": float(np.mean([r["student_censored"] for r in rows])),
    }


def aggregate(spec):
    import numpy as np

    output = Path(spec["output"])
    comparisons, rows = [], []
    for seed in spec["seeds"]:
        folder = output / f"seed_{seed}"
        branches = {b: json.loads((folder / f"{b}.json").read_text())
                    for b in ("student", "teacher", "repeat")}
        checks = {b: compare_snapshots(folder / "student_start.npz", folder / f"{b}_start.npz",
                                      spec["state_tolerance"], spec.get("force_tolerance", 1e-3))
                  for b in ("teacher", "repeat")}
        # Compare the entire sham reward/done/state trajectory, not only totals.
        checks["repeat_trajectory"] = compare_snapshots(
            folder / "student_trace.npz", folder / "repeat_trace.npz", spec["repeat_tolerance"])
        unchanged = [r["match"] for r in branches["student"]["rows"] if not r["eligible"]]
        if unchanged:
            checks["unchanged_matches"] = unchanged_match_check(
                folder / "student_trace.npz", folder / "teacher_trace.npz", unchanged, spec["repeat_tolerance"])
        valid = all(c["passed"] for c in checks.values())
        comparisons.append({"seed": seed, "valid": valid, "checks": checks})
        for s, t, repeat in zip(branches["student"]["rows"], branches["teacher"]["rows"], branches["repeat"]["rows"]):
            row = {"seed": seed, "match": s["match"], "pair_valid": valid,
                   "start_kind": s["start_kind"], "eligible": s["eligible"],
                   "reserved_control": s.get("reserved_control", False),
                   "accepted": s["accepted"], "predicted_advantage": s["predicted_advantage"],
                   "predicted_advantage_finite": s.get("predicted_advantage_finite", True),
                   "real_advantage": t["return"] - s["return"],
                   "repeat_advantage": repeat["return"] - s["return"],
                   "teacher_skill": s["teacher_skill"], "student_skill": s["student_skill"]}
            for b, result in (("student", s), ("teacher", t)):
                for key in ("return", "short_return", "goal", "concession", "failure", "timeout", "censored", "steps"):
                    row[f"{b}_{key}"] = result[key]
            row["short_real_advantage"] = t["short_return"] - s["short_return"]
            rows.append(row)
    with (output / "pairs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    valid_rows = [r for r in rows if r["pair_valid"] and r["eligible"]]
    accepted = [r for r in valid_rows if r["accepted"]]
    report = {
        "status": "paired_checks_passed" if all(c["valid"] for c in comparisons) else "inconclusive_replay_drift",
        "pair_checks": comparisons,
        "all_eligible": summarize(valid_rows), "accepted_teacher": summarize(accepted),
        "accepted_by_start_kind": {str(k): summarize([r for r in accepted if r["start_kind"] == k]) for k in (0, 1, 2)},
        "acceptance_fraction_of_eligible": len(accepted) / max(len(valid_rows), 1),
        "max_abs_repeat_return_difference": float(np.max(np.abs([r["repeat_advantage"] for r in rows]))),
        "interpretation": (
            "Positive real advantage means the teacher first action improved truncated discounted simulator return. "
            "Accepted labels use checkpoint-time readiness gates and the training first-action quality score. "
            "Real branches instead continue the same frozen feedback policy. Censored returns have no learned bootstrap. "
            "This tests teacher actions, not the causal effect of distillation. Separate seeded replays are not an exact "
            "PhysX clone; all exposed tensor states and a student-repeat trajectory are checked. "
            "If replay checks fail, do not interpret the differences as action effects. "
            "Use multiple seed batches; one batch supplies no confidence interval."
        ),
    }
    write_json(output / "summary.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "pair_checks"}, indent=2), flush=True)
    return report


def prepare(args):
    import yaml
    from scripts.record_training_rates import source_data

    run = Path(args.run_dir).resolve()
    config_path = run / "files/config.yaml"
    config = {k: v["value"] for k, v in yaml.safe_load(config_path.read_text()).items()
              if isinstance(v, dict) and "value" in v}
    if "online_mpc" not in config:
        raise ValueError("Expected an online MPC/replay run with saved online_mpc configuration")
    candidates = list((run / "files/tmp/legged_data").glob(f"*/world_model_online_{args.checkpoint}.pt"))
    if len(candidates) != 1:
        raise ValueError(f"Expected one numbered world-model checkpoint, found {candidates}")
    directory = candidates[0].parent
    paths = {"actor": directory / f"ac_weights_{args.checkpoint}.pt",
             "world_model": candidates[0],
             "terminal_value": directory / f"terminal_value_online_{args.checkpoint}.pt",
             "opponent_pool": directory / f"opponent_pool_{args.checkpoint}.pt"}
    history, _ = source_data(str(run))
    metrics = {}
    for row in history:
        iteration = row.get("iterations", -1)
        if iteration > args.checkpoint:
            break
        if iteration >= 0:
            metrics.update({k: v for k, v in row.items()
                            if k.startswith(("mpc_quality/", "mpc/terminal_value", "terminal_value/validation"))})
    if not metrics:
        raise ValueError("No checkpoint-time quality history found")
    spec = vars(args).copy()
    spec.pop("worker_spec", None)
    spec.pop("branch", None)
    spec.pop("seed", None)
    spec["output"] = str(Path(args.output).resolve())
    spec["config"] = config
    spec["quality_metrics"] = metrics
    spec["paths"] = {k: str(p) for k, p in paths.items()}
    spec["sha256"] = {k: sha256(p) for k, p in paths.items()}
    spec["sha256"]["run_config"] = sha256(config_path)
    spec["sha256"]["mpc_yaml"] = sha256(ROOT / config["online_mpc"]["mpc_config"])
    source_paths = [Path(__file__).resolve(), ROOT / "scripts/train_high_level_online_mpc.py",
                    ROOT / "scripts/train_high_level.py", ROOT / "quadruped/mpc/hybrid_cem.py",
                    ROOT / "quadruped/mpc/objective.py", ROOT / "quadruped/mpc/analytical_reward.py",
                    ROOT / "quadruped/envs/wrappers/high_level_skill_wrapper.py",
                    ROOT / "quadruped/envs/wrappers/shared_self_play_wrapper.py",
                    ROOT / config["online_mpc"]["mpc_config"]]
    spec["source_sha256"] = {str(p): sha256(p) for p in source_paths}
    return spec


def worker(spec, seed, branch):
    for path, expected in spec["source_sha256"].items():
        if sha256(path) != expected:
            raise RuntimeError(f"Source changed during evaluation: {path}")
    # Import native Isaac Gym bindings before torch, including in subprocesses.
    import isaacgym  # noqa: F401
    import numpy as np
    import torch
    from types import SimpleNamespace
    from scripts import train_high_level_online_mpc as online
    from scripts.train_high_level import configure_high_level_cfg, load_skill_policies, set_training_seed, resolved_high_level_reward_scales
    from quadruped.envs.base.legged_robot_config import Cfg
    from quadruped.envs.as2.two_robot_velocity_tracking import TwoRobotVelocityTrackingEasyEnv
    from quadruped.envs.wrappers.high_level_skill_wrapper import HighLevelSkillWrapper
    from quadruped.envs.wrappers.shared_self_play_wrapper import SharedPolicySelfPlayWrapper, FrozenOpponentPolicy
    from quadruped_learn.ppo_cse.actor_critic import ActorCritic, AC_Args
    from quadruped.world_model.trainer import load_checkpoint
    from quadruped.world_model.state_adapter import FootballWorldModelStateAdapter
    from quadruped.mpc.config import load_mpc_config
    from quadruped.mpc.objective import MPCObjective
    from quadruped.mpc.hybrid_cem import HybridCEMMPC
    from quadruped.mpc.terminal_value import load_value_checkpoint
    from quadruped.mpc.teacher_quality import improvement_mask
    from quadruped.mpc.simulator_controller import TerminalStateCapture

    def load(path):
        return torch.load(path, map_location=spec["device"])

    def apply_config(target, values):
        for key, value in values.items():
            child = getattr(target, key, None)
            if isinstance(value, dict) and isinstance(child, type):
                apply_config(child, value)
            else:
                setattr(target, key, value)

    config = spec["config"]
    # Parser defaults cover optional fields absent from older saved runs.
    settings = vars(online.build_arg_parser().parse_args([]))
    settings.update(config["online_mpc"])
    settings.update(device=spec["device"], policy_device=spec["device"], num_envs=spec["num_envs"],
                    headless=True, seed=seed, save_video_interval=0, export_step_telemetry=False)
    args = SimpleNamespace(**settings)
    set_training_seed(seed)
    configure_high_level_cfg(Cfg, args)
    apply_config(Cfg, config["Cfg"])
    Cfg.env.num_envs = spec["num_envs"]
    Cfg.env.record_video = False
    Cfg.env.export_step_telemetry = False
    skills = load_skill_policies(args)
    model, model_payload = load_checkpoint(spec["paths"]["world_model"], args.device)
    if model_payload.get("training_config", {}).get("skill_fingerprint") != online.skill_fingerprint(skills):
        raise ValueError("World-model skill provenance does not match the current low-level policies")
    model.eval()
    mpc, _ = load_mpc_config(args.mpc_config, args.mpc_profile)
    mpc.horizon = args.mpc_horizon or mpc.horizon
    mpc.num_candidates = spec["candidates"] or args.mpc_num_samples or mpc.num_candidates
    mpc.num_elites = max(1, min(mpc.num_elites, mpc.num_candidates // 8, mpc.num_candidates - 1))
    mpc.num_iterations = spec["cem_iterations"] or args.mpc_num_iterations or mpc.num_iterations
    online.configure_online_mpc_objective(mpc, args)
    gates = spec["quality_metrics"]
    terminal_active = (gates.get("mpc/terminal_value_active", 0) == 1)
    if spec["terminal_value"] != "checkpoint":
        terminal_active = spec["terminal_value"] == "enabled"
    mpc.use_terminal_value = terminal_active
    mpc.objective_mode = "reward_plus_terminal_value" if terminal_active else "reward_only"
    value = None
    if terminal_active:
        value, value_payload = load_value_checkpoint(spec["paths"]["terminal_value"], args.device)
        if value.schema.to_dict() != model.schema.to_dict() or abs(value_payload["gamma"] - mpc.gamma) > 1e-9:
            raise ValueError("Terminal value schema/gamma mismatch")
    # Keep physics-property randomization identical across value/search ablations.
    set_training_seed(seed)
    raw = TwoRobotVelocityTrackingEasyEnv(sim_device=args.device, headless=True, cfg=Cfg)
    match = HighLevelSkillWrapper(raw, skills)
    env = SharedPolicySelfPlayWrapper(match, args.num_robots, opponent_device=args.device,
                                     opponent_pool_size=args.opponent_pool_size,
                                     opponent_latest_probability=args.opponent_latest_probability)
    adapter = FootballWorldModelStateAdapter(match, max_obstacles=0, num_robots=2 * args.num_robots)
    if adapter.schema.to_dict() != model.schema.to_dict():
        raise ValueError("Simulator/world-model schema mismatch")
    apply_config(AC_Args, config["AC_Args"])
    actor = ActorCritic(env.num_obs, env.num_privileged_obs, env.num_obs_history, env.num_actions).to(args.device)
    actor.load_state_dict(load(spec["paths"]["actor"]))
    actor.eval()
    policy = FrozenOpponentPolicy(actor).to(args.device).eval()
    env.load_opponent_pool_state_dict(load(spec["paths"]["opponent_pool"]), actor)
    # Draw every pool policy on the full batch before selecting its assigned
    # rows. A reset must not change another match's opponent RNG row index.
    def coupled_opponent(observation):
        history = observation["obs_history"]
        selected = torch.zeros(env.match_count, env.team_size, env.num_actions, device=args.device)
        for index, opponent_policy in enumerate(env.opponent_pool):
            sampled = opponent_policy.act_training(history).view_as(selected)
            selected = torch.where((env.opponent_assignment == index)[:, None, None], sampled, selected)
        return selected.reshape(env.num_envs, env.num_actions)
    env.set_opponent_callable(coupled_opponent)

    original_reset = raw.reset_idx
    reset_calls = 0
    reset_devices = [torch.device(args.device).index] if torch.device(args.device).type == "cuda" else []
    def reset_without_rng_leak(env_ids):
        nonlocal reset_calls
        if not len(env_ids):
            return original_reset(env_ids)
        reset_seed = seed + 1000000 + reset_calls
        reset_calls += 1
        return isolated_random_call(lambda: original_reset(env_ids), reset_seed, reset_devices)
    raw.reset_idx = reset_without_rng_leak
    objective = MPCObjective(model.schema, model.action_adapter, model.event_names, mpc,
                             terminal_value=value.predict_with_uncertainty if value else None,
                             controlled_robot_count=args.num_robots,
                             reward_scales=resolved_high_level_reward_scales(args))
    planner = HybridCEMMPC(model, adapter, model.action_adapter, objective, mpc)
    capture = TerminalStateCapture(match, adapter)
    folder = Path(spec["output"]) / f"seed_{seed}"
    folder.mkdir(exist_ok=True)
    team, batch = args.num_robots, args.num_envs

    def policy_action(obs, step):
        # The same draws for the student, opponent and simulator in each branch.
        # Resetting once per decision also isolates CEM's RNG consumption.
        set_training_seed(seed * 100003 + step + 1)
        method = policy.act_training if spec["policy_mode"] == "sample" else policy.act_student
        return method(obs["obs_history"]).detach()

    def snapshot(state, student, opponent, teacher, mask, advantage):
        tensors = {"compact_state": state, "student_action": student, "opponent_action": opponent,
                   "teacher_action": teacher, "intervention_mask": mask, "advantage": advantage}
        # Include low-level histories, joint states, gait phase, commands,
        # randomization tensors, controller state and both teams' observations.
        def collect(prefix, values):
            for key, val in values.items():
                name = prefix + "." + str(key)
                if torch.is_tensor(val):
                    tensors[name] = val
                elif isinstance(val, dict):
                    collect(name, val)
                elif isinstance(val, (list, tuple)):
                    collect(name, dict(enumerate(val)))
        for name, obj in (("raw", raw), ("match", match), ("self_play", env)):
            collect(name, vars(obj))
        np.savez_compressed(folder / f"{branch}_start.npz",
                            **{k: v.detach().cpu().numpy() for k, v in tensors.items()})

    try:
        with torch.no_grad():
            set_training_seed(seed + 1000000)
            obs = env.reset()
            for step in range(spec["warmup_steps"]):
                obs, _, _, _ = env.step(policy_action(obs, step))
            student = policy_action(obs, spec["warmup_steps"])
            state = adapter.extract_state(match)["tensor"]
            opponent = env.preview_opponent_actions().clone()
            canonical = model.action_adapter
            fixed = torch.zeros(batch, mpc.horizon, canonical.action_dim, device=args.device)
            fixed[:, :, 4 * team:] = online._wrapper_actions_to_canonical(opponent, match, canonical)[:, None]
            fixed_mask = torch.zeros(batch, 2 * team, dtype=torch.bool, device=args.device)
            fixed_mask[:, team:] = True
            teacher_mask = torch.ones(batch, team, dtype=torch.bool, device=args.device)
            if args.use_geometric_skill_fallback and args.role_aware_fallback:
                attackers = match._attacker_mask(match._skill_affordances())
                teacher_mask = attackers[:, :team]
                supports = match._walk_support_commands(attacker_mask=attackers)[:, :team]
                fixed.view(batch, mpc.horizon, 2 * team, 4)[:, :, :team] = torch.cat(
                    (torch.zeros_like(supports[..., :1]), supports), -1)[:, None]
                fixed_mask[:, :team] = ~teacher_mask
            own = online._wrapper_actions_to_canonical(student.view(batch, team, 6), match, canonical)
            reference = fixed.clone()
            reference[:, :, :4 * team] = own[:, None]
            plan = planner.plan(state, fixed_action_sequence=fixed, fixed_robot_mask=fixed_mask,
                                reference_action_sequence=reference)
            if args.shooting_options:
                teacher_mask = teacher_mask & (match.shoot_option_remaining[:, :team] <= 0)
            reference = plan.best_action_sequence.clone()
            view = reference[:, 0, :4 * team].reshape(batch, team, 4)
            view[:] = torch.where(teacher_mask[..., None], own.reshape(batch, team, 4), view)
            pair = torch.stack((plan.best_action_sequence, reference), 1)
            # Training distills only the first macro-action from MPC. Score
            # label quality on that same action so open-loop continuation
            # value cannot hide a bad intervention.
            quality_pair = pair[:, :, :1]
            imagined = model.rollout(state, quality_pair, deterministic=True, stop_on_done=False,
                                     action_transform=objective.resolve_imagined_action)
            score = objective.evaluate(state, quality_pair, imagined)
            accepted, _, advantage = improvement_mask(score.total[:, 0], score.total[:, 1],
                                                       score.valid.all(-1), args.mpc_advantage_margin)
            model_ready = (gates.get("mpc_quality/prequential_samples", 0) >= 512
                           and gates.get("mpc_quality/ball_error_m", float("inf")) <= args.mpc_max_ball_error)
            shot_ready = (gates.get("mpc_quality/shot_samples", 0) >= 256
                          and gates.get("mpc_quality/shot_ball_error_m", float("inf")) < .1
                          and gates.get("mpc_quality/shot_velocity_error_mps", float("inf")) < .5)
            chosen_ids, _ = canonical.unpack(plan.first_joint_action)
            accepted &= model_ready
            if args.shooting_options:
                accepted &= ~(chosen_ids[:, :team] == 2).any(-1) | shot_ready
            fallback = plan.uncertainty.get("ood_fallback_used", torch.zeros(batch, dtype=torch.bool, device=args.device))
            accepted &= ~fallback.bool() & teacher_mask.any(-1)
            # Keep an unchanged match in the same vectorized teacher simulation
            # as a control for cross-match RNG/physics coupling.
            reserved = torch.zeros(batch, dtype=torch.bool, device=args.device)
            if spec["control_matches"]:
                reserved[-spec["control_matches"]:] = True
                teacher_mask = teacher_mask & ~reserved[:, None]
                accepted &= ~reserved
            teacher = canonical.to_wrapper_action(plan.first_joint_action).view(batch, 2 * team, 6)[:, :team]
            intervention = torch.where(teacher_mask[..., None], teacher, student.view(batch, team, 6)).reshape_as(student)
            # Nonfinite comparison scores cannot become evidence for a teacher.
            finite_advantage = torch.where(torch.isfinite(advantage), advantage, torch.zeros_like(advantage))
            snapshot(state, student, opponent, intervention, teacher_mask, finite_advantage)
            kinds = getattr(raw, "soccer_start_kind", torch.full((batch,), -1, device=args.device)).clone()
            alive = torch.ones(batch, dtype=torch.bool, device=args.device)
            outcomes = {k: torch.zeros(batch, dtype=torch.bool, device=args.device)
                        for k in ("goal", "concession", "failure", "timeout")}
            trace = {"state": [], "reward": [], "done": [], "actions": []}
            for offset in range(spec["rollout_steps"]):
                action = (intervention if branch == "teacher" else student) if offset == 0 else policy_action(
                    obs, spec["warmup_steps"] + offset)
                capture.clear()
                obs, _, dones, info = env.step(action)
                done = dones.reshape(batch, team).any(-1)
                reward = info["high_level_match_rewards"]
                terminal = alive & done
                for key, field in (("goal", "high_level_goal"), ("concession", "high_level_opponent_goal"),
                                   ("failure", "high_level_learning_team_failure"), ("timeout", "time_outs")):
                    event = torch.as_tensor(info[field], device=args.device).bool()
                    if event.numel() == batch * team:
                        event = event.reshape(batch, team).any(-1)
                    outcomes[key] |= terminal & event.reshape(batch)
                next_state = adapter.extract_state(match)["tensor"]
                next_state = torch.where(capture.valid[:, None], capture.states, next_state)
                trace["state"].append(next_state.cpu().numpy())
                trace["reward"].append(reward.cpu().numpy())
                trace["done"].append(done.cpu().numpy())
                trace["actions"].append(action.cpu().numpy())
                alive &= ~done
                if (offset + 1) % 25 == 0:
                    print(f"{branch}: step {offset + 1}, still active {int(alive.sum())}", flush=True)
            trace = {k: np.stack(v) for k, v in trace.items()}
            measured = real_returns(trace["reward"], trace["done"], mpc.gamma, mpc.horizon)
            rows = []
            for i in range(batch):
                attacker = int(teacher_mask[i].long().argmax())
                rows.append({"match": i, "start_kind": int(kinds[i]), "eligible": bool(teacher_mask[i].any()),
                             "reserved_control": bool(reserved[i]),
                             "accepted": bool(accepted[i]), "predicted_advantage": float(finite_advantage[i]),
                             "predicted_advantage_finite": bool(torch.isfinite(advantage[i])),
                             "teacher_skill": int(chosen_ids[i, attacker]),
                             "student_skill": int(student.view(batch, team, 6)[i, attacker, :3].argmax()),
                             "return": float(measured["return"][i]), "short_return": float(measured["short_return"][i]),
                             "steps": int(measured["steps"][i]), "censored": bool(measured["censored"][i]),
                             **{k: bool(v[i]) for k, v in outcomes.items()}})
            write_json(folder / f"{branch}.json", {"rows": rows, "resolved_mpc": mpc.to_dict(),
                                                    "quality_score_horizon": int(quality_pair.shape[2]),
                                                    "model_ready": model_ready, "shot_ready": shot_ready})
            np.savez_compressed(folder / f"{branch}_trace.npz", **trace)
    finally:
        capture.restore()
        raw.close()
    # Match play_high_level.run_cli: legacy Isaac Gym may segfault during
    # Python/CUDA teardown. Only exit after successful writes and cleanup;
    # exceptions above still propagate and fail the coordinator.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="wandb/run-20260917_011907-32aec9oq")
    parser.add_argument("--checkpoint", type=int, default=800, help="Numbered checkpoint; mutable latest is deliberately unsupported")
    parser.add_argument("--device", default="cuda:4")
    parser.add_argument("--num-envs", type=int, default=1,
                        help="Use one for isolated pairs; vectorized trials require unchanged-match controls")
    parser.add_argument("--control-matches", type=int, default=0,
                        help="Reserve this many final matches for unchanged-action cross-match controls")
    parser.add_argument("--seeds", type=int, nargs="+", default=[50, 51, 52])
    parser.add_argument("--warmup-steps", type=int, default=20, help="Common student prefix before intervention")
    parser.add_argument("--rollout-steps", type=int, default=150, help="Real macro steps, stopping return accumulation at first done")
    parser.add_argument("--policy-mode", choices=["sample", "mode"], default="sample")
    parser.add_argument("--terminal-value", choices=["checkpoint", "enabled", "disabled"], default="checkpoint")
    parser.add_argument("--candidates", type=int, default=None, help="Optional search-budget ablation")
    parser.add_argument("--cem-iterations", type=int, default=None)
    parser.add_argument("--state-tolerance", type=float, default=1e-6)
    parser.add_argument("--force-tolerance", type=float, default=1e-3,
                        help="Separate absolute tolerance for float32 contact-force reductions")
    parser.add_argument("--repeat-tolerance", type=float, default=1e-5)
    parser.add_argument("--output", default="outputs/mpc_teacher_paired_800")
    parser.add_argument("--worker-spec", help=argparse.SUPPRESS)
    parser.add_argument("--branch", choices=["student", "teacher", "repeat"], help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_spec:
        worker(json.loads(Path(args.worker_spec).read_text()), args.seed, args.branch)
        return
    if args.num_envs < 1 or args.rollout_steps < 1 or args.warmup_steps < 0 or args.checkpoint < 0:
        parser.error("Invalid environment/step/checkpoint count")
    if not 0 <= args.control_matches < args.num_envs:
        parser.error("control-matches must be nonnegative and smaller than num-envs")
    if args.num_envs > 1 and args.control_matches == 0:
        parser.error("Vectorized trials require --control-matches 1 (or more) to detect cross-match interference")
    if len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        parser.error("Seeds must be distinct nonnegative integers")
    if args.candidates is not None and args.candidates < 2 or args.cem_iterations is not None and args.cem_iterations < 1:
        parser.error("CEM requires at least two candidates and one iteration")
    if min(args.state_tolerance, args.repeat_tolerance, args.force_tolerance) < 0:
        parser.error("Tolerances must be nonnegative")
    os.chdir(ROOT)
    spec = prepare(args)
    output = Path(spec["output"])
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error("Output directory is nonempty; choose a new directory to preserve previous results")
    manifest = output / "experiment.json"
    write_json(manifest, spec)
    environment = dict(os.environ)
    environment.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/dribblebot_torch_extensions")
    for seed in args.seeds:
        folder = output / f"seed_{seed}"
        folder.mkdir()
        for branch in ("student", "repeat", "teacher"):
            print(f"Running seed {seed}, {branch}; log: {folder / (branch + '.log')}", flush=True)
            command = [sys.executable, str(Path(__file__).resolve()), "--worker-spec", str(manifest),
                       "--seed", str(seed), "--branch", branch]
            with (folder / f"{branch}.log").open("w") as stream:
                subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)
    for key, path in spec["paths"].items():
        if sha256(path) != spec["sha256"][key]:
            raise RuntimeError(f"Checkpoint changed during evaluation: {path}")
    for path, expected in spec["source_sha256"].items():
        if sha256(path) != expected:
            raise RuntimeError(f"Source changed during evaluation: {path}")
    report = aggregate(spec)
    print(f"Results: {output / 'summary.json'}", flush=True)
    if report["status"] != "paired_checks_passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
