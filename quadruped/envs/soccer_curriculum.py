"""Attacking reset curriculum and episode-based progression (no simulator dependency)."""
from collections import deque
import math

import torch

# normal-start share rises by ten percentage points per stage. Retain easy
# finishing examples throughout, rather than replacing them all at promotion.
START_MIX = ((.60, .30), (.50, .30), (.40, .30), (.30, .30), (.25, .25), (.20, .20))


class SoccerCurriculum:
    def __init__(self, min_episodes=512, success_threshold=0.35,
                 opponent_threshold=0.10, max_failure_rate=0.25,
                 min_stage_updates=300, normal_threshold=0.10):
        self.stage = 0
        self.min_episodes = int(min_episodes)
        self.success_threshold = float(success_threshold)
        self.opponent_threshold = float(opponent_threshold)
        self.max_failure_rate = float(max_failure_rate)
        self.min_stage_updates = int(min_stage_updates)
        self.normal_threshold = float(normal_threshold)
        self.stage_updates = 0
        self.windows = [deque(maxlen=self.min_episodes * 2) for _ in range(3)]
        self.opponent_window = deque(maxlen=self.min_episodes * 2)

    @staticmethod
    def lower_bound(values):
        if not values:
            return 0.0
        n = len(values)
        p = sum(v[0] for v in values) / n
        z = 1.96
        return (p + z*z/(2*n) - z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))) / (1+z*z/n)

    def record(self, kinds, stages, goals, failures, anchor=None, concessions=None, timeouts=None):
        anchor = [True]*len(kinds) if anchor is None else anchor
        concessions = [False]*len(kinds) if concessions is None else concessions
        timeouts = [False]*len(kinds) if timeouts is None else timeouts
        for kind, stage, goal, failure, fixed, conceded, timeout in zip(
            kinds, stages, goals, failures, anchor, concessions, timeouts
        ):
            outcome = (bool(goal), bool(failure), bool(conceded), bool(timeout))
            if int(stage) == self.stage:
                self.windows[int(kind)].append(outcome)
            if int(kind) == 0 and fixed:
                self.opponent_window.append(outcome)

    def advance(self):
        # Called once per PPO update, independent of parallel environment count.
        self.stage_updates += 1
        normal_ready = self.stage < 2 or (
            len(self.windows[0]) >= self.min_episodes
            and self.lower_bound(self.windows[0]) >= self.normal_threshold)
        if (self.stage < len(START_MIX) - 1
            and self.stage_updates >= self.min_stage_updates and normal_ready and all(
            len(w) >= self.min_episodes and self.lower_bound(w) >= self.success_threshold
            and sum(v[1] for v in w) / len(w) <= self.max_failure_rate
            for w in self.windows[1:]
        )):
            self.stage += 1
            self.stage_updates = 0
            for w in self.windows:
                w.clear()
            return True
        return False

    def opponent_ready(self):
        w = self.opponent_window
        return (len(w) >= self.min_episodes
                and self.lower_bound(w) >= self.opponent_threshold
                and sum(v[1] for v in w) / len(w) <= self.max_failure_rate)

    def metrics(self):
        finishing, possession = START_MIX[self.stage]
        result = {"curriculum/stage": self.stage,
                  "curriculum/stage_updates": self.stage_updates,
                  "curriculum/normal_start_fraction": 1 - finishing - possession,
                  "curriculum/finishing_start_fraction": finishing,
                  "curriculum/possession_start_fraction": possession}
        for name, window in zip(("normal", "finishing", "possession"), self.windows):
            result[f"curriculum/{name}_episodes"] = len(window)
            result[f"curriculum/{name}_goal_rate"] = sum(v[0] for v in window) / max(len(window), 1)
            result[f"curriculum/{name}_failure_rate"] = sum(v[1] for v in window) / max(len(window), 1)
            result[f"curriculum/{name}_concession_rate"] = sum(v[2] for v in window) / max(len(window), 1)
            result[f"curriculum/{name}_timeout_rate"] = sum(v[3] for v in window) / max(len(window), 1)
        result["curriculum/anchor_normal_episodes"] = len(self.opponent_window)
        result["curriculum/opponent_goal_rate_lower95"] = self.lower_bound(self.opponent_window)
        result["curriculum/opponent_ready"] = float(self.opponent_ready())
        return result

    def state_dict(self):
        return {"version": 2, "stage": self.stage, "stage_updates": self.stage_updates,
                "windows": [list(w) for w in self.windows],
                "opponent_window": list(self.opponent_window)}

    def load_state_dict(self, state):
        self.stage = int(state["stage"])
        legacy = state.get("version", 1) == 1
        if not 0 <= self.stage <= (3 if legacy else len(START_MIX) - 1):
            raise ValueError("Invalid soccer curriculum stage")
        if legacy:
            self.stage = (0, 2, 4, 5)[self.stage]
        self.stage_updates = int(state.get("stage_updates", 0)) if not legacy else 0
        for w, values in zip(self.windows, state["windows"]):
            w.clear()
            if not legacy:
                w.extend(values)
        self.opponent_window.clear()
        self.opponent_window.extend(state.get("opponent_window", []))


def get_curriculum(env):
    if not getattr(env.cfg.env, "soccer_curriculum", False):
        return None
    if not hasattr(env, "soccer_curriculum_state"):
        cfg = env.cfg.env
        env.soccer_curriculum_state = SoccerCurriculum(
            cfg.curriculum_min_episodes, cfg.curriculum_success_threshold,
            cfg.curriculum_opponent_threshold, cfg.curriculum_max_failure_rate,
            getattr(cfg, "curriculum_min_stage_updates", 300),
            getattr(cfg, "curriculum_normal_threshold", .10))
    return env.soccer_curriculum_state


def apply_attacking_starts(env, env_ids, ball_xy):
    curriculum = get_curriculum(env)
    if curriculum is None or len(env_ids) == 0:
        return ball_xy
    if not hasattr(env, "soccer_start_kind"):
        env.soccer_start_kind = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env.soccer_start_stage = torch.zeros_like(env.soccer_start_kind)
    stage = curriculum.stage
    finishing, possession = START_MIX[stage]
    draw = torch.rand(len(env_ids), device=env.device)
    kinds = torch.where(draw < finishing, 1, torch.where(draw < finishing + possession, 2, 0))
    env.soccer_start_kind[env_ids] = kinds
    env.soccer_start_stage[env_ids] = stage
    selected = kinds > 0
    ids = env_ids[selected]
    if len(ids) == 0:
        return ball_xy
    n = len(ids)
    half_length = float(env.cfg.env.field_length) / 2
    half_width = float(env.cfg.env.field_width) / 2
    near = kinds[selected] == 1
    difficulty = stage / (len(START_MIX) - 1)
    # First master close finishing, then approach from farther away. Distances
    # are sampled so a stage does not teach one fixed robot/ball geometry.
    approach_low = torch.where(near, .40 + .10*difficulty, .55 + .65*difficulty)
    approach_high = torch.where(near, .55 + .15*difficulty, .75 + .85*difficulty)
    approach = approach_low + torch.rand(n, device=env.device) * (approach_high-approach_low)
    max_goal_distance = max(1., 2*half_length - (.75 + .85*difficulty) - .65 - .4)
    low = torch.where(near, 1.0 + difficulty, 1.8 + 1.5*difficulty).clamp(max=max_goal_distance)
    high = torch.where(near, 1.6 + 1.9*difficulty, 2.8 + 2*difficulty).clamp(max=max_goal_distance)
    distance = low + torch.rand(n, device=env.device) * (high-low)
    xy = torch.stack((half_length-distance,
                      (torch.rand(n, device=env.device)*2-1) * (.25+.6*difficulty)), dim=-1)
    world = xy + env.env_origins[ids, :2]
    result = ball_xy.clone()
    result[selected] = world
    team = int(env.cfg.env.num_team_robots)
    # Randomize the attacker slot so the shared policy learns both roles.
    attacker = torch.randint(team, (n,), device=env.device)
    for slot in range(env.num_robots):
        actor_ids = env.robot_actor_idxs_all[ids, slot]
        if slot < team:
            is_attacker = attacker == slot
            dx = torch.where(is_attacker, -approach, -approach-.65)
            dy = torch.where(is_attacker, 0., 1.15)
            target = world
        else:
            dx = torch.full((n,), .5 + .45*difficulty, device=env.device)
            dy = torch.full((n,), (1.65-.9*difficulty) * (1 if slot % 2 else -1), device=env.device)
            target = world
        position = xy + torch.stack((dx, dy), dim=-1)
        position[:, 0].clamp_(-half_length+.4, half_length-.4)
        position[:, 1].clamp_(-half_width+.4, half_width-.4)
        position += env.env_origins[ids, :2]
        env.root_states[actor_ids, :2] = position
        delta = target-position
        yaw = torch.atan2(delta[:, 1], delta[:, 0])
        yaw += (torch.rand(n, device=env.device)*2-1) * (.1+.3*difficulty)
        env.root_states[actor_ids, 3:7] = 0
        env.root_states[actor_ids, 5] = torch.sin(yaw/2)
        env.root_states[actor_ids, 6] = torch.cos(yaw/2)
        env.root_states[actor_ids, 7:13] = 0
    return result
