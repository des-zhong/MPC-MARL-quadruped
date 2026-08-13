"""Machine-readable state and action schemas.

All positions are expressed in the team's fixed, global field frame. Planar
positions are divided by the corresponding field half-extent. Linear and
angular velocities remain in m/s and rad/s and are standardized from the
training split. Isaac Gym quaternions use xyzw ordering.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence


@dataclass(frozen=True)
class Feature:
    name: str
    start: int
    size: int
    kind: str
    unit: str
    dynamic: bool = True
    group: str = "other"

    @property
    def stop(self) -> int:
        return self.start + self.size


class StateSchema:
    """Index-safe description of a flat state tensor."""

    def __init__(self, features: Sequence[Feature], version: int = 1):
        self.features = tuple(features)
        self.version = int(version)
        expected = 0
        for feature in self.features:
            if feature.start != expected:
                raise ValueError(f"Non-contiguous schema at {feature.name}: {feature.start} != {expected}")
            expected = feature.stop
        self.state_dim = expected
        self._by_name = {f.name: f for f in self.features}
        if len(self._by_name) != len(self.features):
            raise ValueError("State feature names must be unique")

    def feature(self, name: str) -> Feature:
        try:
            return self._by_name[name]
        except KeyError as exc:
            raise KeyError(f"Unknown state feature {name!r}") from exc

    def slice(self, name: str) -> slice:
        feature = self.feature(name)
        return slice(feature.start, feature.stop)

    def indices(self, *, dynamic: Optional[bool] = None, kinds: Optional[Iterable[str]] = None) -> List[int]:
        allowed = set(kinds) if kinds is not None else None
        result: List[int] = []
        for feature in self.features:
            if dynamic is not None and feature.dynamic != dynamic:
                continue
            if allowed is not None and feature.kind not in allowed:
                continue
            result.extend(range(feature.start, feature.stop))
        return result

    @property
    def continuous_dynamic_indices(self) -> List[int]:
        return self.indices(dynamic=True, kinds=("continuous", "angle_pair"))

    @property
    def binary_dynamic_indices(self) -> List[int]:
        return self.indices(dynamic=True, kinds=("binary",))

    @property
    def static_indices(self) -> List[int]:
        return self.indices(dynamic=False)

    @property
    def yaw_pairs(self) -> List[tuple[int, int]]:
        pairs = []
        for feature in self.features:
            if feature.kind == "angle_pair":
                if feature.size != 2:
                    raise ValueError(f"Angle pair {feature.name} must have size 2")
                pairs.append((feature.start, feature.start + 1))
        return pairs

    def group_positions(self, group: str, within_dynamic: bool = True) -> List[int]:
        absolute = [i for f in self.features if f.group == group for i in range(f.start, f.stop)]
        if not within_dynamic:
            return absolute
        lookup = {absolute_index: i for i, absolute_index in enumerate(self.continuous_dynamic_indices)}
        return [lookup[i] for i in absolute if i in lookup]

    def to_dict(self) -> Dict[str, object]:
        return {"version": self.version, "state_dim": self.state_dim, "features": [asdict(f) for f in self.features]}

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "StateSchema":
        return cls([Feature(**item) for item in payload["features"]], version=int(payload.get("version", 1)))


def default_state_schema(max_obstacles: int = 2, num_robots: int = 2) -> StateSchema:
    """Build the repository's compact multi-robot football state schema."""

    num_robots = int(num_robots)
    max_obstacles = int(max_obstacles)
    if num_robots < 1:
        raise ValueError("num_robots must be at least 1")
    if max_obstacles < 0:
        raise ValueError("max_obstacles must be non-negative")

    features: List[Feature] = []

    def add(name: str, size: int, kind: str, unit: str, dynamic: bool = True, group: str = "other") -> None:
        start = features[-1].stop if features else 0
        features.append(Feature(name, start, size, kind, unit, dynamic, group))

    for robot in range(num_robots):
        prefix = f"robot_{robot}"
        add(f"{prefix}.position", 3, "continuous", "x/y field-half normalized; z m", group="robot_position")
        add(f"{prefix}.yaw_sin_cos", 2, "angle_pair", "unitless", group="robot_orientation")
        add(f"{prefix}.roll_pitch", 2, "continuous", "rad", group="robot_orientation")
        add(f"{prefix}.linear_velocity", 3, "continuous", "m/s", group="robot_velocity")
        add(f"{prefix}.angular_velocity", 3, "continuous", "rad/s", group="robot_velocity")
        add(f"{prefix}.fallen", 1, "binary", "bool", group="robot_event")
        add(f"{prefix}.ball_contact", 1, "binary", "bool", group="robot_event")
        add(f"{prefix}.skill_one_hot", 3, "controlled", "one-hot", group="skill")
        add(f"{prefix}.previous_command", 3, "controlled", "normalized [-1,1]", group="skill")
        add(f"{prefix}.parameter_mask", 3, "controlled", "bool", group="skill")
        add(f"{prefix}.gait_phase_sin_cos", 2, "angle_pair", "unitless cycle encoding", group="phase")
    add("ball.position", 3, "continuous", "x/y field-half normalized; z m", group="ball_position")
    add("ball.linear_velocity", 3, "continuous", "m/s", group="ball_velocity")
    add("ball.angular_velocity", 3, "continuous", "rad/s", group="ball_velocity")
    add("ball.possessed", 1, "binary", "bool", group="ball_event")
    add(
        "ball.possessor_one_hot",
        num_robots + 1,
        "binary",
        "none followed by one entry per robot",
        group="ball_event",
    )
    add("ball.in_opponent_goal", 1, "binary", "bool", group="ball_event")
    add("ball.in_own_goal", 1, "binary", "bool", group="ball_event")
    add("ball.out_of_bounds", 1, "binary", "bool", group="ball_event")
    for obstacle in range(max_obstacles):
        add(
            f"obstacle_{obstacle}.geometry",
            6,
            "static",
            "x/y field-half normalized, half-x m, half-y m, height m, valid bool",
            dynamic=False,
            group="obstacle",
        )
    add("field.geometry", 6, "static", "half-length m, half-width m, goals x m, goal half-width m", False, "field")
    return StateSchema(features)


LEGACY_EVENT_NAMES = (
    "goal",
    "own_goal",
    "out_of_bounds",
    "ball_obstacle_collision",
    "robot_obstacle_collision",
    "teammate_collision",
    "possession_acquired",
    "possession_lost",
    "successful_shot",
    "failed_shot",
)

# Append new labels so the positional meaning of metadata-driven legacy
# datasets remains unchanged.  Dataset readers should continue to use the
# event_names stored in metadata rather than assuming this latest schema.
EVENT_NAMES = LEGACY_EVENT_NAMES + ("pass",)


def validate_event_names(event_names: Sequence[str]) -> tuple[str, ...]:
    """Validate and freeze an ordered event-label schema.

    Event-label arrays are positional, so preserving the order stored in a
    dataset's metadata is essential.  A subset of the current schema is valid
    for backward compatibility (for example, datasets written before the
    ``pass`` label was introduced).
    """

    names = tuple(event_names)
    if not names:
        raise ValueError("Event names cannot be empty")
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("Event names must be non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError("Event names must be unique")
    unknown = [name for name in names if name not in EVENT_NAMES]
    if unknown:
        raise ValueError(f"Unknown event names: {unknown}")
    return names


def event_names_from_metadata(metadata: Mapping[str, object]) -> tuple[str, ...]:
    """Return the ordered event schema stored with a dataset.

    New collection metadata contains the current :data:`EVENT_NAMES`.  Older
    datasets retain their shorter event list, preventing an added output from
    shifting columns or causing a label-width mismatch.
    """

    stored = metadata.get("event_names")
    if stored is None:
        return EVENT_NAMES
    if isinstance(stored, (str, bytes)) or not isinstance(stored, Sequence):
        raise ValueError("metadata['event_names'] must be a sequence of names")
    return validate_event_names(stored)
