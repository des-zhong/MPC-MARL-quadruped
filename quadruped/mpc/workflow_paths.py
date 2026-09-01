"""Conventional paths shared by the model collection/training launchers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


_RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def validate_run_name(name: str) -> str:
    """Reject path-like run names so generated outputs stay inside the project."""

    if not _RUN_NAME.fullmatch(name):
        raise ValueError(
            "Run names may contain only letters, numbers, '.', '_' and '-' "
            "and must start with a letter or number"
        )
    return name


@dataclass(frozen=True)
class ModelWorkflowPaths:
    run: str

    @classmethod
    def for_run(cls, run: str) -> "ModelWorkflowPaths":
        return cls(validate_run_name(run))

    @property
    def initial_dataset(self) -> Path:
        return Path(f"data/world_model_{self.run}")

    @property
    def world_model_output(self) -> Path:
        return Path(f"checkpoints/world_model_{self.run}")

    @property
    def world_model_checkpoint(self) -> Path:
        return self.world_model_output / "best.pt"

    @property
    def bootstrap_value_output(self) -> Path:
        if self.run == "as2":
            return Path("checkpoints/terminal_value_bootstrap")
        return Path(f"checkpoints/terminal_value_{self.run}_bootstrap")

    @property
    def final_value_output(self) -> Path:
        if self.run == "as2":
            return Path("checkpoints/terminal_value_mpc")
        return Path(f"checkpoints/terminal_value_{self.run}_mpc")

    @property
    def working_root(self) -> Path:
        if self.run == "as2":
            return Path("outputs/iterative_mpc_world_model_finetune")
        return Path(f"outputs/model_pipeline_{self.run}")

    @property
    def validation_output(self) -> Path:
        if self.run == "as2":
            return Path("outputs/model_pipeline_validation")
        return Path(f"outputs/model_pipeline_{self.run}_validation")

    def teacher_output(self, iteration: int) -> Path:
        if self.run == "as2":
            return Path(f"data/mpc_teacher_iteration_{iteration:03d}")
        return Path(f"data/mpc_teacher_{self.run}_iteration_{iteration:03d}")

    def expansion_output(self, iteration: int) -> Path:
        if self.run == "as2":
            return Path(f"data/world_model_iterations/mpc_iteration_{iteration:03d}")
        return Path(
            f"data/world_model_iterations/{self.run}_iteration_{iteration:03d}"
        )
