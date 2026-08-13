import csv
import json
import pickle
from collections.abc import Mapping as MappingABC
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset


PathLike = Union[str, Path]
Source = Union[PathLike, Mapping[str, Any], Sequence[Mapping[str, Any]], Sequence[PathLike]]

BINARY_TARGET_KEYS = (
    "success",
    "unsafe",
    "fall",
    "collision",
    "ball_lost",
    "possession_retained",
    "goal",
)
REQUIRED_BINARY_TARGET_KEYS = (
    "success",
    "unsafe",
    "ball_lost",
    "possession_retained",
    "goal",
)


@dataclass
class AffordanceNormalizationStats:
    state_mean: Optional[torch.Tensor] = None
    state_std: Optional[torch.Tensor] = None
    command_mean: Optional[torch.Tensor] = None
    command_std: Optional[torch.Tensor] = None
    duration_mean: Optional[torch.Tensor] = None
    duration_std: Optional[torch.Tensor] = None
    eps: float = 1e-6
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def fit(
        cls,
        tensors: Mapping[str, torch.Tensor],
        normalize_state: bool = True,
        normalize_command: bool = True,
        normalize_duration: bool = True,
        eps: float = 1e-6,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "AffordanceNormalizationStats":
        def moments(value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            value = value.float()
            std = value.std(dim=0, unbiased=False).clamp_min(eps)
            return value.mean(dim=0), std

        stats = cls(eps=eps, metadata=dict(metadata or {}))
        if normalize_state:
            stats.state_mean, stats.state_std = moments(tensors["state_t"])
        if normalize_command:
            stats.command_mean, stats.command_std = moments(tensors["command"])
        if normalize_duration:
            stats.duration_mean, stats.duration_std = moments(tensors["duration"])
        return stats

    def normalize_state(self, value: torch.Tensor) -> torch.Tensor:
        return self._normalize(value, self.state_mean, self.state_std)

    def normalize_command(self, value: torch.Tensor) -> torch.Tensor:
        return self._normalize(value, self.command_mean, self.command_std)

    def normalize_duration(self, value: torch.Tensor) -> torch.Tensor:
        return self._normalize(value, self.duration_mean, self.duration_std)

    @staticmethod
    def _normalize(
        value: torch.Tensor,
        mean: Optional[torch.Tensor],
        std: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if mean is None or std is None:
            return value
        return (value - mean.to(value.device)) / std.to(value.device)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state_mean": self.state_mean,
            "state_std": self.state_std,
            "command_mean": self.command_mean,
            "command_std": self.command_std,
            "duration_mean": self.duration_mean,
            "duration_std": self.duration_std,
            "eps": self.eps,
            "metadata": self.metadata,
        }

    def save(self, path: PathLike) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.to_dict(), path)

    @classmethod
    def load(cls, path: PathLike, map_location: str = "cpu") -> "AffordanceNormalizationStats":
        payload = _torch_load(path, map_location=map_location)
        if isinstance(payload, cls):
            return payload
        if not isinstance(payload, MappingABC):
            raise TypeError(f"Normalization stats at {path} must be a mapping, got {type(payload)}")
        return cls(
            state_mean=payload.get("state_mean"),
            state_std=payload.get("state_std"),
            command_mean=payload.get("command_mean"),
            command_std=payload.get("command_std"),
            duration_mean=payload.get("duration_mean"),
            duration_std=payload.get("duration_std"),
            eps=float(payload.get("eps", 1e-6)),
            metadata=dict(payload.get("metadata", {})),
        )


class AffordanceDataset(Dataset):
    """Rollout sample dataset for skill affordance prediction.

    Each item is a short-horizon execution sample for one robot and one skill.
    The dataset returns nested dictionaries so PyTorch's default collate can
    batch them directly:

    {
        "inputs": {
            "state_t": FloatTensor[state_dim],
            "robot_id": LongTensor[],
            "skill_id": LongTensor[],
            "command": FloatTensor[3],
            "duration": FloatTensor[1],
        },
        "targets": {
            "success": FloatTensor[1],
            ...
            "delta_robot": FloatTensor[2],
            "delta_yaw": FloatTensor[1],
            "delta_ball": FloatTensor[2],
        }
    }
    """

    def __init__(
        self,
        source: Source,
        normalization_path: Optional[PathLike] = None,
        normalizer: Optional[AffordanceNormalizationStats] = None,
        fit_normalization: bool = True,
        normalize_state: bool = True,
        normalize_command: bool = True,
        normalize_duration: bool = True,
        skill_to_id: Optional[Mapping[Any, int]] = None,
        robot_to_id: Optional[Mapping[Any, int]] = None,
        dtype: torch.dtype = torch.float32,
    ):
        loaded_normalizer = None
        if normalizer is None and normalization_path is not None and not fit_normalization:
            loaded_normalizer = AffordanceNormalizationStats.load(normalization_path)
            normalizer = loaded_normalizer

        if skill_to_id is None and loaded_normalizer is not None:
            skill_to_id = loaded_normalizer.metadata.get("skill_to_id")
        if robot_to_id is None and loaded_normalizer is not None:
            robot_to_id = loaded_normalizer.metadata.get("robot_to_id")

        raw = load_affordance_data(source)
        self.skill_to_id = dict(skill_to_id or {})
        self.robot_to_id = dict(robot_to_id or {})
        self.data = _canonicalize_data(
            raw,
            self.skill_to_id,
            self.robot_to_id,
            dtype=dtype,
            allow_new_skill_ids=not bool(self.skill_to_id),
            allow_new_robot_ids=not bool(self.robot_to_id),
        )

        self.skill_to_id = self.data.pop("_skill_to_id")
        self.robot_to_id = self.data.pop("_robot_to_id")
        self.available_binary_targets = tuple(self.data.pop("_available_binary_targets"))
        self.state_metadata = dict(self.data.pop("_state_metadata"))
        self.normalizer = normalizer

        if normalizer is None and (normalize_state or normalize_command or normalize_duration):
            metadata = {
                "skill_to_id": self.skill_to_id,
                "robot_to_id": self.robot_to_id,
                "state_shape": tuple(self.data["state_t"].shape[1:]),
                "state_metadata": self.state_metadata,
            }
            self.normalizer = AffordanceNormalizationStats.fit(
                self.data,
                normalize_state=normalize_state,
                normalize_command=normalize_command,
                normalize_duration=normalize_duration,
                metadata=metadata,
            )
            if normalization_path is not None:
                self.normalizer.save(normalization_path)

    def __len__(self) -> int:
        return int(self.data["state_t"].shape[0])

    def __getitem__(self, index: int) -> Dict[str, Dict[str, torch.Tensor]]:
        state_t = self.data["state_t"][index]
        command = self.data["command"][index]
        duration = self.data["duration"][index]

        if self.normalizer is not None:
            state_t = self.normalizer.normalize_state(state_t)
            command = self.normalizer.normalize_command(command)
            duration = self.normalizer.normalize_duration(duration)

        inputs = {
            "state_t": state_t,
            "robot_id": self.data["robot_id"][index],
            "skill_id": self.data["skill_id"][index],
            "command": command,
            "duration": duration,
        }
        targets = {
            key: self.data[key][index]
            for key in BINARY_TARGET_KEYS
        }
        targets.update({
            "delta_robot": self.data["delta_robot"][index],
            "delta_yaw": self.data["delta_yaw"][index],
            "delta_ball": self.data["delta_ball"][index],
        })
        return {"inputs": inputs, "targets": targets}

    @property
    def state_shape(self) -> Tuple[int, ...]:
        return tuple(self.data["state_t"].shape[1:])

    @property
    def state_dim(self) -> int:
        return int(np.prod(self.state_shape))

    def save_normalization(self, path: PathLike) -> None:
        if self.normalizer is None:
            raise RuntimeError("No normalizer is attached to this dataset.")
        self.normalizer.save(path)


def load_affordance_data(source: Source) -> Dict[str, Any]:
    if isinstance(source, MappingABC):
        return _coerce_payload(source)
    if _is_record_sequence(source):
        return _coerce_payload(source)
    if isinstance(source, (str, Path)):
        return _load_one_file(Path(source))

    sources = list(source)
    if not sources:
        raise ValueError("At least one affordance data source is required.")
    loaded = [_load_one_file(Path(item)) for item in sources]
    return _concat_column_dicts(loaded)


def _torch_load(path: PathLike, map_location: str = "cpu") -> Any:
    try:
        return torch.load(str(Path(path)), map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(str(Path(path)), map_location=map_location)


def _load_one_file(path: Path) -> Dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=True) as payload:
            return {key: payload[key] for key in payload.files}
    if suffix == ".pt":
        return _coerce_payload(_torch_load(path, map_location="cpu"))
    if suffix in (".pkl", ".pickle"):
        with path.open("rb") as file:
            return _coerce_payload(pickle.load(file))
    if suffix == ".csv":
        with path.open("r", newline="") as file:
            return _tabular_records_to_columns(list(csv.DictReader(file)))
    if suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("Reading parquet affordance data requires pandas/pyarrow.") from exc
        return _tabular_records_to_columns(pd.read_parquet(path).to_dict("records"))
    raise ValueError(f"Unsupported affordance data format: {path}")


def _coerce_payload(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, MappingABC):
        if "samples" in payload:
            return _coerce_payload(payload["samples"])
        if "data" in payload and isinstance(payload["data"], (MappingABC, SequenceABC)):
            return _coerce_payload(payload["data"])
        if "inputs" in payload or "targets" in payload:
            merged = {}
            if isinstance(payload.get("inputs"), MappingABC):
                merged.update(payload["inputs"])
            if isinstance(payload.get("targets"), MappingABC):
                merged.update(payload["targets"])
            for key, value in payload.items():
                if key not in ("inputs", "targets"):
                    merged[key] = value
            return merged
        return dict(payload)

    if _is_record_sequence(payload):
        return _records_to_columns(payload)
    raise TypeError(f"Unsupported affordance payload type: {type(payload)}")


def _is_record_sequence(value: Any) -> bool:
    if isinstance(value, (str, bytes, Path, MappingABC)):
        return False
    if not isinstance(value, SequenceABC):
        return False
    return len(value) > 0 and isinstance(value[0], MappingABC)


def _records_to_columns(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    flattened = [_flatten_record(record) for record in records]
    keys = sorted({key for record in flattened for key in record.keys()})
    return {key: [record.get(key) for record in flattened] for key in keys}


def _flatten_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    merged = {}
    if isinstance(record.get("inputs"), MappingABC):
        merged.update(record["inputs"])
    if isinstance(record.get("targets"), MappingABC):
        merged.update(record["targets"])
    for key, value in record.items():
        if key not in ("inputs", "targets"):
            merged[key] = value
    return merged


def _tabular_records_to_columns(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not records:
        raise ValueError("Tabular affordance data is empty.")

    out: Dict[str, Any] = {
        "command": _read_vector(
            records,
            ("command", "cmd"),
            ("command_", "cmd_"),
            alias_columns=(("cmd_x", "cmd_y", "cmd_yaw"), ("command_x", "command_y", "command_yaw")),
        ),
        "duration": _read_scalar(records, ("duration", "horizon_s", "dt")),
        "delta_robot": _read_vector(
            records,
            ("delta_robot",),
            ("delta_robot_",),
            alias_columns=(("dx", "dy"), ("delta_robot_x", "delta_robot_y")),
        ),
        "delta_yaw": _read_scalar(records, ("delta_yaw", "dyaw")),
        "delta_ball": _read_vector(
            records,
            ("delta_ball",),
            ("delta_ball_",),
            alias_columns=(("dx_ball", "dy_ball"), ("delta_ball_x", "delta_ball_y")),
        ),
    }

    state_t = _read_optional_vector(records, ("state_t", "state"), ("state_t_", "state_"))
    if state_t is not None:
        out["state_t"] = state_t
    else:
        robot_states = _read_robot_states(records)
        ball_state = _read_optional_vector(
            records,
            ("ball_state", "ball"),
            ("ball_state_", "ball_"),
            alias_columns=(("ball_x", "ball_y", "ball_vx", "ball_vy"),),
        )
        if robot_states is None or ball_state is None:
            raise KeyError(
                "Tabular affordance data must provide either state_t/state columns "
                "or structured robot_*_state plus ball_state columns."
            )
        out["robot_states"] = robot_states
        out["ball_state"] = ball_state

    if _has_any_column(records, ("robot_id", "robot")):
        out["robot_id"] = _read_scalar(records, ("robot_id", "robot"))
    if _has_any_column(records, ("skill_id", "skill_name", "skill")):
        key = _first_column(records, ("skill_id", "skill_name", "skill"))
        out[key] = [record[key] for record in records]

    for key in BINARY_TARGET_KEYS:
        if _has_any_column(records, (key,)):
            out[key] = _read_scalar(records, (key,))
    return out


def _read_vector(
    records: Sequence[Mapping[str, Any]],
    exact_names: Sequence[str],
    prefixes: Sequence[str],
    alias_columns: Sequence[Sequence[str]] = (),
) -> np.ndarray:
    first = records[0]
    for name in exact_names:
        if name in first:
            return np.asarray([_parse_vector(record[name]) for record in records], dtype=np.float32)

    fieldnames = list(first.keys())
    for aliases in alias_columns:
        if all(alias in first for alias in aliases):
            return np.asarray([[float(record[alias]) for alias in aliases] for record in records], dtype=np.float32)

    for prefix in prefixes:
        columns = _prefixed_columns(fieldnames, prefix)
        if columns:
            return np.asarray([[float(record[column]) for column in columns] for record in records], dtype=np.float32)

    raise KeyError(f"Missing vector columns: {exact_names} or prefixes {prefixes}")


def _read_optional_vector(
    records: Sequence[Mapping[str, Any]],
    exact_names: Sequence[str],
    prefixes: Sequence[str],
    alias_columns: Sequence[Sequence[str]] = (),
) -> Optional[np.ndarray]:
    try:
        return _read_vector(records, exact_names, prefixes, alias_columns)
    except KeyError:
        return None


def _read_robot_states(records: Sequence[Mapping[str, Any]]) -> Optional[np.ndarray]:
    first = records[0]
    exact_names = ("robot_states", "robots_state")
    for name in exact_names:
        if name in first:
            return np.asarray([_parse_robot_states(record[name]) for record in records], dtype=np.float32)

    robot_vectors = []
    robot_index = 0
    while True:
        aliases = (
            (f"robot{robot_index}_x", f"robot{robot_index}_y", f"robot{robot_index}_yaw"),
            (f"robot_{robot_index}_x", f"robot_{robot_index}_y", f"robot_{robot_index}_yaw"),
        )
        vector = _read_optional_vector(
            records,
            (f"robot{robot_index}_state", f"robot_{robot_index}_state"),
            (f"robot{robot_index}_state_", f"robot_{robot_index}_state_"),
            alias_columns=aliases,
        )
        if vector is None:
            break
        robot_vectors.append(vector)
        robot_index += 1

    if not robot_vectors:
        return None
    return np.stack(robot_vectors, axis=1)


def _read_scalar(records: Sequence[Mapping[str, Any]], names: Sequence[str]) -> List[Any]:
    name = _first_column(records, names)
    return [record[name] for record in records]


def _has_any_column(records: Sequence[Mapping[str, Any]], names: Sequence[str]) -> bool:
    first = records[0]
    return any(name in first for name in names)


def _first_column(records: Sequence[Mapping[str, Any]], names: Sequence[str]) -> str:
    first = records[0]
    for name in names:
        if name in first:
            return name
    raise KeyError(f"Missing required column, expected one of {names}")


def _prefixed_columns(fieldnames: Sequence[str], prefix: str) -> List[str]:
    columns = [name for name in fieldnames if name.startswith(prefix)]

    def suffix_index(name: str) -> int:
        suffix = name[len(prefix):]
        if suffix.isdigit():
            return int(suffix)
        return 10_000

    return sorted(columns, key=lambda name: (suffix_index(name), name))


def _parse_vector(value: Any) -> List[float]:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float().flatten().tolist()
    if isinstance(value, np.ndarray):
        return value.astype(np.float32).reshape(-1).tolist()
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]

    text = str(value).strip()
    if text.startswith("["):
        return [float(item) for item in json.loads(text)]
    text = text.replace(";", " ").replace(",", " ")
    return [float(item) for item in text.split() if item]


def _parse_robot_states(value: Any) -> List[List[float]]:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().float().numpy()
    elif isinstance(value, np.ndarray):
        array = value.astype(np.float32)
    elif isinstance(value, (list, tuple)):
        array = np.asarray(value, dtype=np.float32)
    else:
        array = np.asarray(json.loads(str(value)), dtype=np.float32)

    if array.ndim != 2:
        raise ValueError(f"robot_states must have shape [num_robots, robot_state_dim], got {array.shape}")
    return array.tolist()


def _canonicalize_data(
    data: Mapping[str, Any],
    skill_to_id: Mapping[Any, int],
    robot_to_id: Mapping[Any, int],
    dtype: torch.dtype,
    allow_new_skill_ids: bool,
    allow_new_robot_ids: bool,
) -> Dict[str, torch.Tensor]:
    data = dict(data)
    command = _as_width_tensor(_find_vector(data, ("command", "cmd"), (("cmd_x", "cmd_y", "cmd_yaw"),)), 3, "command", dtype)
    num_samples = command.shape[0]

    state_t, state_metadata = _build_state_t(data, num_samples, dtype)
    duration = _as_width_tensor(_require_any(data, ("duration",)), 1, "duration", dtype, num_samples)

    robot_id_values = _find_optional_vector(data, ("robot_id", "robot"), ())
    if robot_id_values is None:
        robot_id_values = np.zeros(num_samples, dtype=np.int64)
    robot_id, robot_map = _encode_id_vector(
        robot_id_values,
        robot_to_id,
        "robot_id",
        num_samples,
        allow_new_ids=allow_new_robot_ids,
    )

    skill_values = _find_optional_vector(data, ("skill_id", "skill_name", "skill"), ())
    if skill_values is None:
        raise KeyError("Missing skill_id or skill_name.")
    skill_id, skill_map = _encode_id_vector(
        skill_values,
        skill_to_id,
        "skill_id",
        num_samples,
        allow_new_ids=allow_new_skill_ids,
    )

    out: Dict[str, torch.Tensor] = {
        "state_t": state_t,
        "robot_id": robot_id,
        "skill_id": skill_id,
        "command": command,
        "duration": duration,
    }

    for key in REQUIRED_BINARY_TARGET_KEYS:
        out[key] = _as_binary_target(_require_any(data, (key,)), key, num_samples)
    for key in ("fall", "collision"):
        if key in data:
            out[key] = _as_binary_target(data[key], key, num_samples)
        else:
            out[key] = torch.zeros(num_samples, 1, dtype=dtype)

    out["delta_robot"] = _as_width_tensor(
        _find_vector(data, ("delta_robot",), (("dx", "dy"), ("delta_robot_x", "delta_robot_y"))),
        2,
        "delta_robot",
        dtype,
        num_samples,
    )
    out["delta_yaw"] = _as_width_tensor(
        _find_vector(data, ("delta_yaw", "dyaw"), ()),
        1,
        "delta_yaw",
        dtype,
        num_samples,
    )
    out["delta_ball"] = _as_width_tensor(
        _find_vector(data, ("delta_ball",), (("dx_ball", "dy_ball"), ("delta_ball_x", "delta_ball_y"))),
        2,
        "delta_ball",
        dtype,
        num_samples,
    )
    out["_skill_to_id"] = skill_map
    out["_robot_to_id"] = robot_map
    out["_available_binary_targets"] = tuple(key for key in BINARY_TARGET_KEYS if key in data)
    out["_state_metadata"] = state_metadata
    return out


def _build_state_t(
    data: Mapping[str, Any],
    num_samples: int,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    if "state_t" in data or "state" in data:
        state_t = _as_state_tensor(_require_any(data, ("state_t", "state")), num_samples, dtype)
        return state_t, {
            "format": "flat",
            "state_shape": tuple(state_t.shape[1:]),
        }

    if "robot_states" not in data or "ball_state" not in data:
        raise KeyError(
            "Missing state. Provide either state_t/state, or structured "
            "robot_states [N, num_robots, robot_state_dim] and ball_state [N, ball_state_dim]."
        )

    robot_states = _as_robot_states(data["robot_states"], num_samples)
    ball_state = _as_state_tensor(data["ball_state"], num_samples, dtype=torch.float32)
    state_parts = [
        robot_states.reshape(num_samples, -1),
        ball_state.detach().cpu().numpy().reshape(num_samples, -1),
    ]
    opponent_states = None
    for key in ("opponent_states", "static_opponent_states", "obstacle_states"):
        if key in data:
            opponent_states = _as_entity_states(data[key], num_samples, key)
            state_parts.append(opponent_states.reshape(num_samples, -1))
            break

    state_array = np.concatenate(state_parts, axis=1)
    state_t = torch.as_tensor(state_array, dtype=dtype)
    metadata = {
        "format": "structured_two_quadruped_ball",
        "num_robots": int(robot_states.shape[1]),
        "robot_state_dim": int(robot_states.shape[2]),
        "ball_state_dim": int(ball_state.shape[1]),
        "state_shape": tuple(state_t.shape[1:]),
        "layout": "flatten(robot_states) followed by ball_state",
    }
    if opponent_states is not None:
        metadata.update(
            {
                "format": "structured_two_quadruped_ball_static_opponents",
                "num_static_opponents": int(opponent_states.shape[1]),
                "opponent_state_dim": int(opponent_states.shape[2]),
                "layout": "flatten(robot_states), ball_state, then flatten(opponent_states)",
            }
        )
    return state_t, metadata


def _require_any(data: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    raise KeyError(f"Missing required field, expected one of {keys}")


def _find_vector(
    data: Mapping[str, Any],
    keys: Sequence[str],
    alias_columns: Sequence[Sequence[str]],
) -> Any:
    value = _find_optional_vector(data, keys, alias_columns)
    if value is None:
        raise KeyError(f"Missing required vector field, expected one of {keys}")
    return value


def _find_optional_vector(
    data: Mapping[str, Any],
    keys: Sequence[str],
    alias_columns: Sequence[Sequence[str]],
) -> Optional[Any]:
    for key in keys:
        if key in data:
            return data[key]
    for aliases in alias_columns:
        if all(alias in data for alias in aliases):
            return np.stack([_to_numpy(data[alias]).reshape(-1) for alias in aliases], axis=1)
    return None


def _as_state_tensor(value: Any, num_samples: int, dtype: torch.dtype) -> torch.Tensor:
    array = _to_numpy(value).astype(np.float32)
    if array.ndim == 1:
        if num_samples == 1:
            array = array.reshape(1, -1)
        elif array.shape[0] == num_samples:
            array = array.reshape(num_samples, 1)
        else:
            raise ValueError(f"state_t has shape {array.shape}, expected first dimension {num_samples}.")
    if array.shape[0] != num_samples:
        raise ValueError(f"state_t has shape {array.shape}, expected first dimension {num_samples}.")
    return torch.as_tensor(array, dtype=dtype)


def _as_robot_states(value: Any, num_samples: int) -> np.ndarray:
    array = _as_entity_states(value, num_samples, "robot_states")
    if array.shape[1] != 2:
        raise ValueError(
            f"This affordance dataset is configured for exactly 2 quadrupeds, got {array.shape[1]} robots."
        )
    return array


def _as_entity_states(value: Any, num_samples: int, name: str) -> np.ndarray:
    array = _to_numpy(value).astype(np.float32)
    if array.ndim == 2 and num_samples == 1:
        array = array.reshape(1, array.shape[0], array.shape[1])
    if array.ndim != 3:
        raise ValueError(f"{name} must have shape [N, num_entities, state_dim], got {array.shape}.")
    if array.shape[0] != num_samples:
        raise ValueError(f"{name} has {array.shape[0]} samples, expected {num_samples}.")
    return array


def _as_width_tensor(
    value: Any,
    width: int,
    name: str,
    dtype: torch.dtype,
    num_samples: Optional[int] = None,
) -> torch.Tensor:
    array = _to_numpy(value).astype(np.float32)
    if array.ndim == 0:
        array = array.reshape(1, 1)
    elif array.ndim == 1:
        if width == 1:
            array = array.reshape(-1, 1)
        elif array.shape[0] == width:
            array = array.reshape(1, width)
        elif num_samples is not None and array.shape[0] == num_samples * width:
            array = array.reshape(num_samples, width)
        else:
            raise ValueError(f"{name} has shape {array.shape}, expected width {width}.")

    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"{name} has shape {array.shape}, expected [N, {width}].")
    if num_samples is not None:
        if array.shape[0] == 1 and num_samples > 1:
            array = np.repeat(array, num_samples, axis=0)
        elif array.shape[0] != num_samples:
            raise ValueError(f"{name} has {array.shape[0]} samples, expected {num_samples}.")
    return torch.as_tensor(array, dtype=dtype)


def _as_binary_target(value: Any, name: str, num_samples: int) -> torch.Tensor:
    values = _to_bool_float_array(value)
    tensor = _as_width_tensor(values, 1, name, torch.float32, num_samples)
    return tensor.clamp(0.0, 1.0)


def _to_bool_float_array(value: Any) -> np.ndarray:
    array = _to_numpy(value)
    if array.dtype.kind in ("U", "S", "O"):
        mapping = {
            "1": 1.0,
            "true": 1.0,
            "yes": 1.0,
            "y": 1.0,
            "0": 0.0,
            "false": 0.0,
            "no": 0.0,
            "n": 0.0,
            "": 0.0,
            "none": 0.0,
        }
        flat = []
        for item in array.reshape(-1):
            text = str(item).strip().lower()
            if text not in mapping:
                raise ValueError(f"Cannot parse binary value: {item}")
            flat.append(mapping[text])
        return np.asarray(flat, dtype=np.float32).reshape(array.shape)
    return array.astype(np.float32)


def _encode_id_vector(
    value: Any,
    provided_map: Mapping[Any, int],
    name: str,
    num_samples: int,
    allow_new_ids: bool,
) -> Tuple[torch.Tensor, Dict[Any, int]]:
    array = _to_numpy(value)
    if array.ndim == 0:
        array = np.repeat(array.reshape(1), num_samples)
    array = array.reshape(-1)
    if array.shape[0] == 1 and num_samples > 1:
        array = np.repeat(array, num_samples)
    if array.shape[0] != num_samples:
        raise ValueError(f"{name} has {array.shape[0]} ids, expected {num_samples}.")

    if array.dtype.kind in ("U", "S", "O"):
        texts = [str(item).strip() for item in array]
        if not provided_map and all(_is_int_text(text) for text in texts):
            return torch.as_tensor([int(text) for text in texts], dtype=torch.long), {}

        id_map = dict(provided_map or {})
        encoded = []
        next_id = max(id_map.values(), default=-1) + 1
        for key in texts:
            if key not in id_map:
                if not allow_new_ids:
                    raise KeyError(f"Unknown {name} value {key!r}; pass the same id mapping used for training.")
                id_map[key] = next_id
                next_id += 1
            encoded.append(id_map[key])
        return torch.as_tensor(encoded, dtype=torch.long), id_map

    return torch.as_tensor(array.astype(np.int64), dtype=torch.long), dict(provided_map or {})


def _is_int_text(text: str) -> bool:
    if text.startswith("-"):
        text = text[1:]
    return text.isdigit()


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, list) and value and isinstance(value[0], torch.Tensor):
        return torch.stack([item.detach().cpu() for item in value]).numpy()
    if isinstance(value, list) and value and isinstance(value[0], np.ndarray):
        return np.stack(value, axis=0)
    return np.asarray(value)


def _concat_column_dicts(items: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if len(items) == 1:
        return dict(items[0])
    keys = sorted(set().union(*(item.keys() for item in items)))
    out = {}
    for key in keys:
        values = [item[key] for item in items if key in item]
        if len(values) != len(items):
            raise KeyError(f"Field {key} is missing from at least one source.")
        out[key] = np.concatenate([_to_numpy(value) for value in values], axis=0)
    return out
