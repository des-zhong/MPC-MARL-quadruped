"""Episode diagnostics; no early termination or timeout reclassification."""
import torch


class AttackDiagnostics:
    def __init__(self, count, device):
        self.close = torch.zeros(count, dtype=torch.bool, device=device)
        self.shot = torch.zeros_like(self.close)
        self.launch = torch.zeros_like(self.close)
        self.unrecovered_slow_ball = torch.zeros_like(self.close)
        self.initial_distance = torch.full((count,), float('nan'), device=device)
        self.best_distance = self.initial_distance.clone()

    def observe(self, ball_distance, robot_ball_distance, shooting, goalward_speed, active=None):
        active = torch.ones_like(self.close) if active is None else active
        initial = torch.isnan(self.initial_distance) & active
        self.initial_distance[initial] = ball_distance[initial]
        self.best_distance[initial] = ball_distance[initial]
        self.best_distance = torch.where(active, torch.minimum(self.best_distance, ball_distance), self.best_distance)
        self.close |= (robot_ball_distance <= .8) & active
        self.shot |= shooting & (robot_ball_distance <= .8) & active
        self.launch |= shooting & (robot_ball_distance <= 1.) & (goalward_speed >= .8) & active
        self.unrecovered_slow_ball = torch.where(
            active, self.launch & (goalward_speed < .2) & (robot_ball_distance > .8),
            self.unrecovered_slow_ball)

    def finish(self, done, timeout):
        result = {
            'attack/timeout_no_contact': timeout & ~self.close,
            'attack/timeout_no_shot': timeout & self.close & ~self.shot,
            'attack/timeout_shot_no_launch': timeout & self.close & self.shot & ~self.launch,
            'attack/timeout_launch_no_goal': timeout & self.close & self.shot & self.launch,
            'attack/timeout_after_launch_slow_unrecovered': timeout & self.unrecovered_slow_ball,
            'attack/episode_reached_ball': done & self.close,
            'attack/episode_shot_requested_near_ball': done & self.shot,
            'attack/episode_goalward_launch': done & self.launch,
            'attack/episode_count': done,
            'attack/timeout_no_goal_progress': timeout & ((self.initial_distance-self.best_distance) < .25),
            'attack/episode_goal_progress_m': torch.where(done, torch.nan_to_num(self.initial_distance-self.best_distance), torch.zeros_like(self.best_distance)),
        }
        self.close[done] = False
        self.shot[done] = False
        self.launch[done] = False
        self.unrecovered_slow_ball[done] = False
        self.initial_distance[done] = float('nan')
        self.best_distance[done] = float('nan')
        return result
