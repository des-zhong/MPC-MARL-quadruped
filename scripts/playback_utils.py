"""Robot-neutral helpers shared by policy playback scripts."""

import hashlib
import math
import os
import re
import shutil
import tempfile
import warnings
from pathlib import Path
from urllib.parse import urlparse


GAITS = {
    "pronking": [0.0, 0.0, 0.0],
    "trotting": [0.5, 0.0, 0.0],
    "bounding": [0.0, 0.5, 0.0],
    "pacing": [0.0, 0.0, 0.5],
}

DEFAULT_POLICY_ACTION_CLIP = 1.0


def normalize_wandb_run_path(wandb_run):
    if wandb_run.startswith("http://") or wandb_run.startswith("https://"):
        parsed = urlparse(wandb_run)
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) >= 4 and parts[2] == "runs":
            return f"{parts[0]}/{parts[1]}/{parts[3]}"
        raise ValueError(f"Could not parse W&B run URL: {wandb_run}")
    return wandb_run.replace("/runs/", "/").strip("/")


def _safe_path_component(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._") or "run"


def wandb_run_cache_dir(run_path, cache_root=None):
    """Return a stable cache directory unique to one W&B run.

    W&B policy files from different runs commonly have identical remote names
    (for example ``tmp/legged_data/body_latest.jit``).  Restoring those names
    into a shared root lets a later restore overwrite an earlier policy while
    callers still hold the same path.  The digest keeps sanitized run names
    collision-free.
    """

    normalized = normalize_wandb_run_path(run_path)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]
    label = "--".join(_safe_path_component(part) for part in normalized.split("/"))
    if cache_root is None:
        configured_root = os.environ.get("DRIBBLEBOT_WANDB_CACHE_DIR")
        cache_root = (
            Path(configured_root).expanduser()
            if configured_root
            else Path(__file__).resolve().parents[1] / "tmp" / "wandb_restore_cache"
        )
    return Path(cache_root) / f"{label}--{digest}"


def _cache_candidate_path(run_cache, candidate):
    relative = Path(candidate)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"W&B restore candidate must be a safe relative path, got {candidate!r}")
    return run_cache / relative


def _env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def restore_wandb_file(run_path, candidates, cache_root=None, refresh=None):
    """Restore the first available file into a run-scoped local cache.

    Cache reuse is the default so a resolved ``latest`` policy remains pinned
    and offline playback works.  Pass ``refresh=True`` (or set
    ``DRIBBLEBOT_WANDB_REFRESH=1``) to re-query mutable W&B files.  If a refresh
    cannot reach W&B, a pre-existing run-scoped file remains a safe fallback.
    """

    import wandb

    run_path = normalize_wandb_run_path(run_path)
    run_cache = wandb_run_cache_dir(run_path, cache_root).expanduser().resolve()
    run_cache.mkdir(parents=True, exist_ok=True)
    if refresh is None:
        refresh = _env_flag("DRIBBLEBOT_WANDB_REFRESH")
    refresh = bool(refresh)
    errors = []
    for candidate in candidates:
        cached_path = _cache_candidate_path(run_cache, candidate)
        cached_exists = cached_path.is_file()
        if cached_exists and not refresh:
            return cached_path.resolve()

        # Refresh through a sibling staging directory so an interrupted W&B
        # download cannot partially overwrite the known-good cached file.
        refresh_tmp = (
            tempfile.TemporaryDirectory(prefix=".refresh-", dir=str(run_cache))
            if refresh
            else None
        )
        restore_root = Path(refresh_tmp.name) if refresh_tmp is not None else run_cache
        try:
            try:
                restored = wandb.restore(
                    candidate,
                    run_path=run_path,
                    root=str(restore_root),
                    replace=refresh,
                )
            except Exception as exc:
                if cached_exists:
                    warnings.warn(
                        f"Could not refresh {candidate!r} from W&B run {run_path}; "
                        f"using cached file {cached_path}: {exc}",
                        RuntimeWarning,
                    )
                    return cached_path.resolve()
                errors.append(f"{candidate}: {exc}")
                continue

            if restored is not None:
                # Current W&B versions preserve the remote subdirectory below
                # ``root``.  The copy fallback also supports older versions
                # that return a path outside the requested root.
                restored_close = getattr(restored, "close", None)
                if callable(restored_close):
                    restored_close()
                restored_path = Path(restored.name)
                if not restored_path.is_absolute():
                    restored_path = restored_path.resolve()

                expected_restored_path = _cache_candidate_path(restore_root, candidate)
                source_path = (
                    expected_restored_path
                    if expected_restored_path.is_file()
                    else restored_path
                )
                if source_path.is_file():
                    cached_path.parent.mkdir(parents=True, exist_ok=True)
                    if refresh:
                        # Ensure the source is on the cache filesystem before
                        # os.replace provides the atomic commit.
                        if restore_root not in source_path.parents:
                            staged_path = _cache_candidate_path(restore_root, candidate)
                            staged_path.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(str(source_path), str(staged_path))
                            source_path = staged_path
                        os.replace(str(source_path), str(cached_path))
                    elif source_path != cached_path:
                        shutil.copy2(str(source_path), str(cached_path))
                    return cached_path.resolve()
                errors.append(f"{candidate}: restored path {restored_path} does not exist")
                continue
            if cached_exists:
                warnings.warn(
                    f"W&B returned no refreshed file for {candidate!r} in run {run_path}; "
                    f"using cached file {cached_path}",
                    RuntimeWarning,
                )
                return cached_path.resolve()
            errors.append(f"{candidate}: wandb.restore returned None")
        finally:
            if refresh_tmp is not None:
                refresh_tmp.cleanup()

    raise FileNotFoundError("Could not restore from W&B:\n" + "\n".join(errors))


def file_sha256(path, chunk_size=1024 * 1024):
    """Compute a streaming SHA-256 checksum for a checkpoint artifact."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def find_policy_config_path(body_path, search_roots=()):
    """Find the training ``config.yaml`` adjacent to a local/cached policy."""

    body_path = Path(body_path)
    candidate_roots = [body_path.parent]
    # Python 3.8's pathlib parents sequence supports integer indexing but not
    # slicing (slice support was added later).
    candidate_roots.extend(list(body_path.parents)[1:6])
    candidate_roots.extend(Path(root) for root in search_roots if root is not None)
    seen = set()
    for root in candidate_roots:
        for candidate in (root / "config.yaml", root / "files" / "config.yaml"):
            key = str(candidate.resolve(strict=False))
            if key in seen:
                continue
            seen.add(key)
            if candidate.is_file():
                return candidate.resolve()
    return None


def find_local_wandb_config(run_path, search_root=None):
    """Find an already-downloaded W&B config by immutable run ID."""

    normalized = normalize_wandb_run_path(run_path)
    run_id = normalized.rsplit("/", 1)[-1]
    root = Path(search_root) if search_root is not None else Path(__file__).resolve().parents[1] / "wandb"
    if not root.is_dir():
        return None
    candidates = sorted(root.glob(f"*{run_id}*/files/config.yaml"))
    if not candidates:
        return None
    # Run IDs are unique; prefer the newest local copy if multiple archived
    # directories happen to contain the same run.
    return max(candidates, key=lambda path: path.stat().st_mtime).resolve()


def _unwrap_wandb_config_value(value):
    if isinstance(value, dict) and "value" in value and len(value) <= 2:
        return value["value"]
    return value


def policy_action_clip_from_config(config_path):
    """Read ``Cfg.normalization.clip_actions`` from a run config."""

    import yaml

    with Path(config_path).open("rb") as file:
        payload = yaml.safe_load(file) or {}
    cfg = _unwrap_wandb_config_value(payload.get("Cfg", payload))
    if not isinstance(cfg, dict):
        raise ValueError(f"Cfg is not a mapping in {config_path}")
    normalization = _unwrap_wandb_config_value(cfg.get("normalization", {}))
    if not isinstance(normalization, dict) or "clip_actions" not in normalization:
        raise KeyError(f"Cfg.normalization.clip_actions is missing from {config_path}")
    clip = float(_unwrap_wandb_config_value(normalization["clip_actions"]))
    if not math.isfinite(clip) or clip <= 0.0:
        raise ValueError(f"Invalid action clip {clip!r} in {config_path}")
    return clip


def build_policy_metadata(
    body_path,
    adaptation_module_path,
    ac_weights_path=None,
    *,
    run_path=None,
    checkpoint=None,
    config_path=None,
    action_clip=None,
    fallback_action_clip=DEFAULT_POLICY_ACTION_CLIP,
):
    """Build auditable checkpoint identity and action-range metadata."""

    body_path = Path(body_path).resolve()
    adaptation_module_path = Path(adaptation_module_path).resolve()
    ac_weights_path = Path(ac_weights_path).resolve() if ac_weights_path is not None else None
    if config_path is None:
        config_path = find_policy_config_path(body_path)
    elif config_path is not None:
        config_path = Path(config_path).resolve()

    if action_clip is not None:
        resolved_clip = float(action_clip)
        clip_source = "explicit"
    elif config_path is not None:
        try:
            resolved_clip = policy_action_clip_from_config(config_path)
            clip_source = f"config:{config_path}"
        except (KeyError, TypeError, ValueError):
            resolved_clip = float(fallback_action_clip)
            clip_source = f"fallback:{fallback_action_clip} (invalid config:{config_path})"
    else:
        resolved_clip = float(fallback_action_clip)
        clip_source = f"fallback:{fallback_action_clip} (config unavailable)"
    if not math.isfinite(resolved_clip) or resolved_clip <= 0.0:
        raise ValueError(f"Policy action clip must be finite and positive, got {resolved_clip!r}")

    def artifact(path):
        if path is None:
            return None
        return {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }

    return {
        "run_path": normalize_wandb_run_path(run_path) if run_path else None,
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "config_path": str(config_path) if config_path is not None else None,
        "action_clip": resolved_clip,
        "action_clip_source": clip_source,
        "artifacts": {
            "body": artifact(body_path),
            "adaptation_module": artifact(adaptation_module_path),
            "ac_weights": artifact(ac_weights_path),
            "config": artifact(config_path),
        },
    }


def resolve_policy_files(run_dir, checkpoint, body_path=None, adaptation_module_path=None):
    if body_path and adaptation_module_path:
        return Path(body_path), Path(adaptation_module_path)

    root = Path(run_dir)
    candidate_dirs = [
        root,
        root / "tmp" / "legged_data",
        root / "files" / "tmp" / "legged_data",
    ]
    candidate_dirs += [path.parent for path in root.glob("**/body_latest.jit")]

    valid_dirs = []
    for candidate in candidate_dirs:
        if candidate.exists() and candidate.is_dir() and candidate not in valid_dirs:
            valid_dirs.append(candidate)

    def files_for(directory, name):
        body = directory / f"body_{name}.jit"
        adaptation = directory / f"adaptation_module_{name}.jit"
        if body.exists() and adaptation.exists():
            return body, adaptation
        return None

    if checkpoint != "latest":
        for directory in valid_dirs:
            found = files_for(directory, checkpoint)
            if found:
                return found
        raise FileNotFoundError(
            f"Could not find body_{checkpoint}.jit and adaptation_module_{checkpoint}.jit under {root}"
        )

    latest_candidates = []
    for directory in valid_dirs:
        found = files_for(directory, "latest")
        if found:
            latest_candidates.append(found)
    if latest_candidates:
        return max(latest_candidates, key=lambda pair: pair[0].stat().st_mtime)

    numbered = []
    for directory in valid_dirs:
        for body in directory.glob("body_*.jit"):
            suffix = body.stem[len("body_") :]
            if not suffix.isdigit():
                continue
            adaptation = directory / f"adaptation_module_{suffix}.jit"
            if adaptation.exists():
                numbered.append((int(suffix), body, adaptation))
    if numbered:
        _, body, adaptation = max(numbered, key=lambda item: item[0])
        return body, adaptation

    raise FileNotFoundError(f"Could not find JIT policy files under {root}")


def resolve_ac_weights_file(
    run_dir,
    checkpoint,
    ac_weights_path=None,
    policy_dir=None,
):
    if ac_weights_path:
        return Path(ac_weights_path)

    suffix = "latest" if checkpoint in ("latest", "last") else checkpoint
    candidate_dirs = []
    if policy_dir is not None:
        candidate_dirs.append(Path(policy_dir))

    root = Path(run_dir)
    candidate_dirs += [
        root,
        root / "tmp" / "legged_data",
        root / "files" / "tmp" / "legged_data",
    ]

    candidates = []
    for directory in candidate_dirs:
        candidates.append(directory / f"ac_weights_{suffix}.pt")
        if suffix == "latest":
            candidates.append(directory / "ac_weights_last.pt")

    existing = [candidate for candidate in candidates if candidate.exists()]
    if existing:
        return max(existing, key=lambda path: path.stat().st_mtime)
    return None


def load_ac_weights(ac_weights_path, map_location):
    import torch

    if ac_weights_path is None:
        print("No actor-critic .pt checkpoint found; using JIT modules for inference.")
        return None

    checkpoint = torch.load(str(ac_weights_path), map_location=map_location)
    if isinstance(checkpoint, dict):
        print(f"Loaded actor-critic checkpoint: {ac_weights_path} ({len(checkpoint)} tensors)")
    else:
        print(f"Loaded actor-critic checkpoint: {ac_weights_path} ({type(checkpoint).__name__})")
    return checkpoint


def get_raw_env(env):
    # Keep Isaac Gym imports out of this robot-neutral helper's module import;
    # playback entry points load Isaac Gym before torch as required.
    from dribblebot.envs.wrappers.history_wrapper import HistoryWrapper

    if isinstance(env, HistoryWrapper):
        return env.env
    return env


def get_sensor_slice(raw_env, sensor_class_name):
    start = 0
    for sensor in raw_env.sensors:
        dim = sensor.get_dim()
        if sensor.__class__.__name__ == sensor_class_name:
            return slice(start, start + dim)
        start += dim
    raise ValueError(f"Sensor {sensor_class_name} was not found")


def set_walking_command(raw_env, xyz_yaw_cmd, args):
    import torch

    raw_env.commands[:, :] = 0.0
    raw_env.commands[:, 0] = float(xyz_yaw_cmd[0])
    raw_env.commands[:, 1] = float(xyz_yaw_cmd[1])
    raw_env.commands[:, 2] = float(xyz_yaw_cmd[2])

    if raw_env.cfg.commands.num_commands > 3:
        raw_env.commands[:, 3] = args.body_height
    if raw_env.cfg.commands.num_commands > 4:
        raw_env.commands[:, 4] = args.step_frequency
    if raw_env.cfg.commands.num_commands > 7:
        gait = torch.tensor(GAITS[args.gait], dtype=raw_env.commands.dtype, device=raw_env.device)
        raw_env.commands[:, 5:8] = gait
    if raw_env.cfg.commands.num_commands > 8:
        raw_env.commands[:, 8] = args.gait_duration
    if raw_env.cfg.commands.num_commands > 9:
        raw_env.commands[:, 9] = args.footswing_height
    if raw_env.cfg.commands.num_commands > 10:
        raw_env.commands[:, 10] = args.pitch
    if raw_env.cfg.commands.num_commands > 11:
        raw_env.commands[:, 11] = args.roll
    if raw_env.cfg.commands.num_commands > 12:
        raw_env.commands[:, 12] = args.stance_width
    if raw_env.cfg.commands.num_commands > 13:
        raw_env.commands[:, 13] = args.stance_length
    if raw_env.cfg.commands.num_commands > 14:
        raw_env.commands[:, 14] = args.aux_reward_coef


def patch_obs_command(raw_env, obs, command_slice):
    scaled_commands = raw_env.commands * raw_env.commands_scale
    obs["obs"][:, command_slice] = scaled_commands
    history_start = obs["obs_history"].shape[1] - raw_env.num_obs
    obs["obs_history"][
        :,
        history_start + command_slice.start : history_start + command_slice.stop,
    ] = scaled_commands
