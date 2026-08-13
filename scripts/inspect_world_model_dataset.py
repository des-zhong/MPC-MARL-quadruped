"""Generate data-quality diagnostics before world-model training."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from dribblebot.world_model.dataset import assert_no_episode_leakage, load_manifest


def main(args) -> None:
    root = Path(args.dataset)
    metadata = json.loads((root / "metadata.json").read_text())
    manifests = {name: load_manifest(root, name) for name in ("train", "validation", "test")}
    split_ids = {name: [int(item["episode_id"]) for item in manifest["episodes"]] for name, manifest in manifests.items()}
    assert_no_episode_leakage(split_ids)
    entries = load_manifest(root)["episodes"]
    num_robots = int(metadata.get("num_robots", metadata.get("action_schema", {}).get("num_robots", 2)))
    lengths, rewards, skills, events, sources = [], [], [Counter() for _ in range(num_robots)], None, Counter()
    targeted_scenarios = Counter()
    termination_reasons = Counter()
    modified_actions = invalid_skill_requests = audited_actions = total_actions = 0
    command_values = [[[] for _ in range(3)] for _ in range(num_robots)]
    termination_causes = Counter()
    state_min = state_max = None
    warnings = []
    fingerprints = set()
    duplicates = 0
    feature_indices = {
        feature["name"]: int(feature["start"])
        for feature in metadata.get("state_schema", {}).get("features", [])
    }
    terminal_robot_falls = Counter()
    for entry in entries:
        with np.load(root / entry["path"], allow_pickle=False) as shard:
            state = shard["state"]
            action = shard["joint_action"]
            event = shard["event_labels"]
            lengths.append(len(state))
            rewards.extend(shard["reward"].tolist())
            for robot in range(num_robots):
                offset = 4 * robot
                skill_ids = action[:, offset].astype(int)
                skills[robot].update(skill_ids.tolist())
                if not np.isin(skill_ids, [0, 1, 2]).all(): warnings.append(f"impossible skill ID in episode {entry['episode_id']}")
                for skill_id in range(3):
                    selected = action[skill_ids == skill_id, offset + 1 : offset + 4]
                    if len(selected):
                        command_values[robot][skill_id].append(selected)
                        bounds = metadata["action_schema"]["bounds"][str(skill_id)]
                        low, high = np.asarray(bounds["low"]), np.asarray(bounds["high"])
                        mask = np.asarray(bounds["mask"])
                        if (selected < low - 1e-6).any() or (selected > high + 1e-6).any() or np.abs(selected * (1 - mask)).max() > 1e-6:
                            warnings.append(f"action outside bounds in episode {entry['episode_id']}")
            sources.update(shard["behavior_source"].astype(str).tolist())
            if "targeted_scenario" in shard:
                targeted_scenarios.update(
                    value for value in shard["targeted_scenario"].astype(str).tolist() if value
                )
            if "termination_reason" in shard:
                termination_reasons.update(
                    value for value in shard["termination_reason"].astype(str).tolist() if value != "none"
                )
            if "requested_action_modified" in shard:
                modified_actions += int(shard["requested_action_modified"].sum())
                audited_actions += len(state)
            if "invalid_skill_requested" in shard:
                invalid_skill_requests += int(shard["invalid_skill_requested"].sum())
            total_actions += len(state)
            events = event.sum(0) if events is None else events + event.sum(0)
            state_min = state.min(0) if state_min is None else np.minimum(state_min, state.min(0))
            state_max = state.max(0) if state_max is None else np.maximum(state_max, state.max(0))
            if not np.isfinite(state).all() or not np.isfinite(shard["next_state"]).all(): warnings.append(f"NaN/Inf in episode {entry['episode_id']}")
            if len(state) > 1 and not np.allclose(shard["next_state"][:-1], state[1:], atol=1e-5):
                warnings.append(f"state discontinuity in episode {entry['episode_id']}")
            termination_causes["terminated"] += int(shard["terminated"].sum())
            termination_causes["truncated"] += int(shard["truncated"].sum())
            terminal = np.asarray(shard["terminated"]).astype(bool)
            fall_names = [f"robot_{robot}.fallen" for robot in range(num_robots)]
            if terminal.any() and all(name in feature_indices for name in fall_names):
                terminal_next = shard["next_state"][terminal]
                fallen = np.stack(
                    [terminal_next[:, feature_indices[name]] > 0.5 for name in fall_names],
                    axis=1,
                )
                for robot in range(num_robots):
                    terminal_robot_falls[f"robot_{robot}"] += int(fallen[:, robot].sum())
                terminal_robot_falls["any_robot"] += int(fallen.any(axis=1).sum())
                terminal_robot_falls["all_robots"] += int(fallen.all(axis=1).sum())
            for row in np.concatenate((state, action), axis=1):
                fingerprint = hash(row.tobytes())
                duplicates += fingerprint in fingerprints
                fingerprints.add(fingerprint)
    constant = np.flatnonzero(np.isclose(state_min, state_max)).tolist()
    event_names = metadata["event_names"]
    event_counts = {name: int(events[index]) for index, name in enumerate(event_names)}
    if constant: warnings.append(f"constant state feature indices: {constant}")
    if any(sum(counter.values()) and min(counter.values(), default=0) < 0.05 * sum(counter.values()) for counter in skills): warnings.append("severe skill imbalance")
    absent = [name for name, count in event_counts.items() if count == 0]
    if absent: warnings.append(f"absent rare events: {absent}")
    minimum_counts = metadata.get("minimum_event_counts")
    if minimum_counts is None:
        minimum_counts = (
            metadata.get("config", {})
            .get("data_collection", {})
            .get("minimum_event_counts", {})
        )
    coverage = {
        name: {
            "count": int(event_counts.get(name, 0)),
            "minimum": int(minimum),
            "met": int(event_counts.get(name, 0)) >= int(minimum),
        }
        for name, minimum in minimum_counts.items()
    }
    below_minimum = [name for name, values in coverage.items() if not values["met"]]
    if below_minimum:
        warnings.append(f"rare-event coverage below configured minimum: {below_minimum}")
    if lengths and float(np.mean(lengths)) < 5.0:
        warnings.append("mean episode length is below 5 macro steps; inspect early robot falls and policy loading")
    command_report = []
    for robot in range(num_robots):
        by_skill = {}
        for skill_id in range(3):
            if command_values[robot][skill_id]:
                values = np.concatenate(command_values[robot][skill_id], axis=0)
                by_skill[str(skill_id)] = {"min": values.min(0).tolist(), "mean": values.mean(0).tolist(), "max": values.max(0).tolist()}
            else:
                by_skill[str(skill_id)] = {"count": 0}
        command_report.append(by_skill)
    successful = event_counts.get("successful_shot", 0)
    failed = event_counts.get("failed_shot", 0)
    report = {
        "episodes": len(entries), "transitions": int(sum(lengths)),
        "episode_length": {"min": int(min(lengths)), "mean": float(np.mean(lengths)), "max": int(max(lengths))},
        "skill_frequency": [{str(k): v for k, v in counter.items()} for counter in skills],
        "command_parameters_by_skill": command_report,
        "state_feature_ranges": {str(index): [float(state_min[index]), float(state_max[index])] for index in range(len(state_min))},
        "reward": {"min": float(np.min(rewards)), "mean": float(np.mean(rewards)), "max": float(np.max(rewards))},
        "termination_causes": dict(termination_causes), "event_counts": event_counts,
        "termination_reasons": dict(termination_reasons),
        "terminal_robot_falls": dict(terminal_robot_falls),
        "event_coverage": coverage,
        "successful_shot_rate": float(successful / max(successful + failed, 1)),
        "behavior_source": dict(sources),
        "targeted_scenario": dict(targeted_scenarios),
        "action_execution": {
            "transitions": total_actions,
            "audited_transitions": audited_actions,
            "available": bool(audited_actions),
            "requested_action_modified": modified_actions,
            "requested_action_modified_rate": (
                float(modified_actions / audited_actions) if audited_actions else None
            ),
            "invalid_skill_requested": invalid_skill_requests,
            "invalid_skill_requested_rate": (
                float(invalid_skill_requests / audited_actions) if audited_actions else None
            ),
        },
        "duplicate_transition_count": duplicates,
        "splits": {name: len(ids) for name, ids in split_ids.items()}, "warnings": warnings,
    }
    output = Path(args.output or root / "dataset_report.json")
    output.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/world_model_as2")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
