"""Episode-preserving evaluation export from frozen PPO simulator rollouts."""

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from quadruped.mpc.simulator_controller import TerminalStateCapture, validate_environment_compatibility
from quadruped.mpc.teacher_dataset import episode_arrays
from .action_adapter import JointActionAdapter
from .dataset import Episode, EpisodeShardWriter
from .schema import StateSchema
from .state_adapter import FootballWorldModelStateAdapter


class EvaluationRolloutRecorder:
    def __init__(self, env, args, policy, skill_policies=None):
        self.env = env
        self.match = env.env
        self.output = Path(args.world_model_output)
        if self.output.exists() and any(self.output.iterdir()):
            raise FileExistsError(f"Refusing to overwrite nonempty collection directory: {self.output}")
        payload = torch.load(args.world_model_checkpoint, map_location="cpu")
        if payload.get("format") != "dribblebot_world_model_v1":
            raise ValueError("Expected a world-model checkpoint")
        schema = StateSchema.from_dict(payload["state_schema"])
        self.actions = JointActionAdapter.from_dict(payload["action_schema"])
        obstacles = sum(f.group == "obstacle" for f in schema.features)
        self.adapter = FootballWorldModelStateAdapter(
            self.match, obstacles, num_robots=self.actions.num_robots,
            event_names=payload["event_names"])
        if self.adapter.schema.to_dict() != schema.to_dict():
            raise ValueError("PPO environment state schema differs from world-model checkpoint; "
                             "use a PPO checkpoint with matching robot count and shooting options")
        validate_environment_compatibility(
            self.match, SimpleNamespace(action_adapter=self.actions), payload)
        self.event_names = list(payload["event_names"])
        self.count = 0
        self.target = int(args.stop_after_episodes)
        self.buffers = [[] for _ in range(args.num_envs)]
        self.per_env = [0] * args.num_envs
        self.metadata = {
            "state_schema": schema.to_dict(), "action_schema": self.actions.to_dict(),
            "event_names": self.event_names, "purpose": "held_out_ppo_evaluation",
            "high_level_policy_dir": str(Path(args.high_level_policy_dir).resolve()),
            "high_level_checkpoint": args.high_level_checkpoint,
            "opponent_checkpoint": args.opponent_high_level_checkpoint,
            "world_model_checkpoint": str(Path(args.world_model_checkpoint).resolve()),
            "policy_source": policy.get("source"), "seed": args.seed,
            "policy_metadata": policy.get("policy_metadata", {}),
            "skill_policy_metadata": {name: record.get("policy_metadata", {})
                                      for name, record in (skill_policies or {}).items()},
            "collection_args": vars(args).copy(),
            "macro_action_steps": int(self.match.control_interval),
            "high_level_dt": float(self.match.env.dt * self.match.control_interval),
            "requested_episodes": self.target * args.num_envs,
        }
        self.writer = EpisodeShardWriter(self.output, self.metadata)
        self.capture = TerminalStateCapture(self.match, self.adapter)

    def before_step(self):
        self.capture.clear()
        self.state = self.adapter.extract_state(self.match)["tensor"].detach().clone()

    def after_step(self, done, info):
        device = self.state.device
        batch = len(self.buffers)
        match_done = done.reshape(batch, -1).any(-1).bool()
        timeout = torch.as_tensor(info["time_outs"], device=device).reshape(batch, -1).any(-1).bool()
        if bool((match_done & ~self.capture.valid).any()):
            raise RuntimeError("Terminal state was not captured before reset")
        live = self.adapter.extract_state(self.match)["tensor"]
        next_state = torch.where(self.capture.valid[:, None], self.capture.states, live)
        actions = self.actions.pack(
            torch.as_tensor(info["high_level_skill_ids"], device=device, dtype=torch.long),
            torch.as_tensor(info["high_level_commands"], device=device, dtype=torch.float))
        self.actions.assert_within_bounds(actions, atol=1e-4)
        rewards = torch.as_tensor(info["high_level_match_rewards"], device=device).reshape(batch, -1)[:, 0]
        elapsed = torch.as_tensor(info["elapsed_low_level_steps"], device=device).reshape(batch)
        events = self.adapter.extract_event_labels(self.state, next_state, info)
        fields = {"state": self.state, "joint_action": actions, "reward": rewards,
                  "next_state": next_state, "terminated": match_done & ~timeout,
                  "truncated": match_done & timeout, "elapsed_low_level_steps": elapsed,
                  "event_labels": events}
        arrays = {key: value.detach().cpu().numpy() for key, value in fields.items()}
        for index, buffer in enumerate(self.buffers):
            if self.per_env[index] >= self.target:
                continue
            record = {key: value[index].copy() for key, value in arrays.items()}
            record["behavior_source"] = "ppo_evaluation"
            buffer.append(record)
            if bool(match_done[index]):
                self.writer.write_episode(Episode(self.count, episode_arrays(self.count, buffer)))
                self.count += 1
                self.per_env[index] += 1
                buffer.clear()
                self._write_test_manifest()
                print(f"Saved evaluation episode {self.count}/{self.metadata['requested_episodes']}", flush=True)

    def _write_test_manifest(self):
        manifest = json.loads((self.output / "manifest.json").read_text())
        manifest["split"] = "test"
        (self.output / "test_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    def close(self):
        self.capture.restore()

    def verify_complete(self):
        if self.count != self.metadata["requested_episodes"]:
            raise RuntimeError(f"Collected {self.count}/{self.metadata['requested_episodes']} complete episodes. "
                               "Increase --steps. Partial episodes were not exported.")
