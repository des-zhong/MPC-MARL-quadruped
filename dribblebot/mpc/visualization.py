"""Headless tactical, CEM, and prediction-vs-reality visualizations."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np
import torch


SKILL_LABELS = ("reposition", "dribble", "shoot")


def _num_robots(schema) -> int:
    return sum(feature.name.startswith("robot_") and feature.name.endswith(".position") for feature in schema.features)


def _numpy(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _field_coordinates(states, schema):
    states = _numpy(states)
    field = states[..., schema.slice("field.geometry")]
    scale = field[..., :2]
    robot = np.stack(
        [
            states[..., schema.slice(f"robot_{index}.position")][..., :2] * scale
            for index in range(_num_robots(schema))
        ],
        axis=-2,
    )
    ball = states[..., schema.slice("ball.position")][..., :2] * scale
    return field, robot, ball


def plot_top_down(
    schema,
    current_state,
    predicted_states=None,
    action_sequence=None,
    uncertainty=None,
    real_history=None,
    actual_future=None,
    output: Optional[Union[str, Path]] = None,
    title: str = "MPC tactical view",
    terminal_value: Optional[float] = None,
    controlled_robot_count: Optional[int] = None,
):
    """Plot field, real history, predicted plan, and optional actual future."""

    current = _numpy(current_state).reshape(-1)
    field = current[schema.slice("field.geometry")]
    half_length, half_width = float(field[0]), float(field[1])
    figure, axis = plt.subplots(figsize=(11, 7))
    axis.add_patch(
        Rectangle(
            (-half_length, -half_width),
            2 * half_length,
            2 * half_width,
            fill=False,
            color="black",
            linewidth=2,
            label="field",
        )
    )
    goal_half = float(field[4])
    for goal_x, label, hatch in (
        (float(field[2]), "own goal", "//"),
        (float(field[3]), "opponent goal", "\\\\"),
    ):
        axis.plot(
            [goal_x, goal_x],
            [-goal_half, goal_half],
            color="black",
            linewidth=5,
            linestyle="-",
            label=label,
        )
        axis.add_patch(
            Rectangle(
                (min(goal_x, goal_x - 0.15), -goal_half),
                0.15,
                2 * goal_half,
                fill=False,
                hatch=hatch,
                edgecolor="gray",
            )
        )
    for feature in schema.features:
        if feature.group != "obstacle":
            continue
        geometry = current[feature.start : feature.stop]
        if geometry[5] <= 0.5:
            continue
        center = geometry[:2] * field[:2]
        radius = float(np.linalg.norm(geometry[2:4]))
        axis.add_patch(
            Circle(
                center,
                radius,
                facecolor="none",
                edgecolor="tab:gray",
                linestyle=":",
                linewidth=2,
                hatch="xx",
                label="static box (conservative radius)",
            )
        )

    _, current_robot, current_ball = _field_coordinates(current[None], schema)
    num_robots = _num_robots(schema)
    marker_choices = ("^", "s", "P", "X", "D", "v", "<", ">")
    markers = [marker_choices[index % len(marker_choices)] for index in range(num_robots)]
    if controlled_robot_count is None:
        colors = [plt.get_cmap("tab10")(index % 10) for index in range(num_robots)]
    else:
        controlled_robot_count = int(controlled_robot_count)
        learning_colors = ("tab:blue", "tab:cyan", "navy", "deepskyblue")
        opponent_colors = ("tab:red", "firebrick", "lightcoral", "darkred")
        colors = [
            (
                learning_colors[index % len(learning_colors)]
                if index < controlled_robot_count
                else opponent_colors[
                    (index - controlled_robot_count) % len(opponent_colors)
                ]
            )
            for index in range(num_robots)
        ]

    def robot_label(index, suffix):
        if controlled_robot_count is None:
            return f"robot {index} {suffix}"
        if index < controlled_robot_count:
            return f"learning {index} {suffix}"
        return f"opponent {index - controlled_robot_count} {suffix}"

    for robot in range(num_robots):
        xy = current_robot[0, robot]
        axis.scatter(
            xy[0],
            xy[1],
            marker=markers[robot],
            s=100,
            color=colors[robot],
            edgecolor="black",
            label=robot_label(robot, "current"),
            zorder=6,
        )
        yaw_pair = current[schema.slice(f"robot_{robot}.yaw_sin_cos")]
        axis.arrow(
            xy[0],
            xy[1],
            0.35 * yaw_pair[1],
            0.35 * yaw_pair[0],
            width=0.025,
            color=colors[robot],
            length_includes_head=True,
        )
    axis.scatter(
        current_ball[0, 0],
        current_ball[0, 1],
        marker="o",
        s=90,
        color="white",
        edgecolor="black",
        label="ball current",
        zorder=7,
    )

    if real_history is not None:
        _, history_robot, history_ball = _field_coordinates(real_history, schema)
        for robot in range(num_robots):
            axis.plot(
                history_robot[:, robot, 0],
                history_robot[:, robot, 1],
                color=colors[robot],
                linestyle="-",
                marker=markers[robot],
                markevery=max(1, len(history_robot) // 8),
                linewidth=2,
                label=robot_label(robot, "real history"),
            )
        axis.plot(
            history_ball[:, 0],
            history_ball[:, 1],
            color="black",
            linestyle="-",
            marker="o",
            markevery=max(1, len(history_ball) // 8),
            label="ball real history",
        )

    if predicted_states is not None:
        _, predicted_robot, predicted_ball = _field_coordinates(
            predicted_states, schema
        )
        uncertainty_np = _numpy(uncertainty)
        for robot in range(num_robots):
            axis.plot(
                predicted_robot[:, robot, 0],
                predicted_robot[:, robot, 1],
                color=colors[robot],
                linestyle="--",
                marker=markers[robot],
                linewidth=2,
                label=robot_label(robot, "predicted"),
            )
        axis.plot(
            predicted_ball[:, 0],
            predicted_ball[:, 1],
            color="black",
            linestyle="--",
            marker="x",
            linewidth=2,
            label="ball predicted",
        )
        terminal_label = "predicted terminal state"
        if terminal_value is not None:
            terminal_label += f"; V(s_H)={float(terminal_value):.3f}"
        axis.scatter(
            predicted_ball[-1, 0], predicted_ball[-1, 1], marker="*", s=230,
            color="tab:orange", edgecolor="black", zorder=9, label=terminal_label,
        )
        if uncertainty_np is not None and len(predicted_ball) > 1:
            scaled = 30 + 170 * (
                uncertainty_np
                / max(float(np.nanmax(uncertainty_np)), 1.0e-8)
            )
            axis.scatter(
                predicted_ball[1:, 0],
                predicted_ball[1:, 1],
                s=scaled,
                facecolors="none",
                edgecolors="tab:red",
                marker="o",
                label="plan uncertainty",
            )
        if action_sequence is not None:
            actions = _numpy(action_sequence).reshape(-1, num_robots, 4)
            for step in range(min(len(actions), len(predicted_ball) - 1)):
                label = "/".join(
                    SKILL_LABELS[int(skill)]
                    for skill in actions[step, :, 0]
                )
                axis.annotate(
                    f"{step}: {label}",
                    predicted_ball[step + 1],
                    xytext=(4, 5),
                    textcoords="offset points",
                    fontsize=7,
                )

    if actual_future is not None:
        _, actual_robot, actual_ball = _field_coordinates(actual_future, schema)
        for robot in range(num_robots):
            axis.plot(
                actual_robot[:, robot, 0],
                actual_robot[:, robot, 1],
                color=colors[robot],
                linestyle="-.",
                marker="P",
                linewidth=2,
                label=robot_label(robot, "actual future"),
            )
        axis.plot(
            actual_ball[:, 0],
            actual_ball[:, 1],
            color="tab:green",
            linestyle="-.",
            marker="D",
            label="ball actual future",
        )

    axis.set(
        xlim=(-half_length - 0.5, half_length + 0.5),
        ylim=(-half_width - 0.5, half_width + 0.5),
        xlabel="field x (m)",
        ylabel="field y (m)",
        title=title,
        aspect="equal",
    )
    axis.grid(True, linestyle=":", alpha=0.3)
    handles, labels = axis.get_legend_handles_labels()
    unique = {}
    for handle, label in zip(handles, labels):
        unique.setdefault(label, handle)
    axis.legend(unique.values(), unique.keys(), loc="upper center", ncol=3, fontsize=8)
    figure.tight_layout()
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=160)
    return figure


def plot_candidate_endpoints(
    schema,
    current_state,
    final_states,
    objectives,
    elite_count: int,
    output: Union[str, Path],
    predicted_returns=None,
    terminal_values=None,
    terminal_uncertainty=None,
):
    final_states = _numpy(final_states)
    objectives = _numpy(objectives)
    _, robots, ball = _field_coordinates(final_states, schema)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    scatter = axes[0].scatter(
        ball[:, 0],
        ball[:, 1],
        c=objectives,
        cmap="viridis",
        marker="o",
        s=25,
    )
    elite = np.argsort(objectives)[-min(elite_count, len(objectives)) :]
    best = int(np.argmax(objectives))
    predicted_returns = _numpy(predicted_returns)
    terminal_values = _numpy(terminal_values)
    terminal_uncertainty = _numpy(terminal_uncertainty)
    axes[0].scatter(
        ball[elite, 0],
        ball[elite, 1],
        facecolors="none",
        edgecolors="black",
        marker="s",
        s=70,
        label="elite endpoints",
    )
    axes[0].scatter(
        ball[best, 0],
        ball[best, 1],
        color="red",
        marker="*",
        s=180,
        label="best endpoint",
    )
    detail = ""
    if predicted_returns is not None and terminal_values is not None:
        detail = f"\nbest: R_H={predicted_returns[best]:.2f}, V-term={terminal_values[best]:.2f}"
        if terminal_uncertainty is not None:
            detail += f", U_H={terminal_uncertainty[best]:.2f}"
    axes[0].set(title="Candidate ball endpoints" + detail, xlabel="x (m)", ylabel="y (m)")
    axes[0].legend()
    figure.colorbar(scatter, ax=axes[0], label="objective")
    marker_choices = ("^", "s", "P", "X", "D", "v", "<", ">")
    for robot in range(robots.shape[1]):
        axes[1].scatter(
            robots[:, robot, 0],
            robots[:, robot, 1],
            c=objectives,
            cmap="plasma",
            marker=marker_choices[robot % len(marker_choices)],
            s=25,
            label=f"robot {robot}",
        )
    axes[1].set(title="Candidate robot endpoints", xlabel="x (m)", ylabel="y (m)")
    axes[1].legend()
    for axis in axes:
        axis.grid(True, linestyle=":", alpha=0.3)
        axis.set_aspect("equal")
    figure.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def plot_cem_convergence(convergence: Mapping[str, object], output):
    best = _numpy(convergence["best_objective"])
    elite = _numpy(convergence["mean_elite_objective"])
    objective_std = _numpy(convergence["elite_objective_std"])
    entropy = _numpy(convergence["skill_entropy"])
    parameter_std = _numpy(convergence["mean_parameter_std"])
    first_probs = _numpy(convergence["first_step_skill_probabilities"])
    iterations = np.arange(best.shape[-1])
    figure, axes = plt.subplots(3, 2, figsize=(12, 11))
    axes[0, 0].plot(iterations, best, marker="o", label="best")
    axes[0, 0].plot(iterations, elite, marker="s", linestyle="--", label="elite mean")
    axes[0, 0].fill_between(
        iterations, elite - objective_std, elite + objective_std, alpha=0.2
    )
    axes[0, 0].set_title("CEM objective")
    axes[0, 0].legend()
    axes[0, 1].plot(iterations, entropy, marker="o")
    axes[0, 1].set_title("Categorical skill entropy")
    axes[1, 0].plot(iterations, parameter_std, marker="s")
    axes[1, 0].set_title("Mean normalized parameter std")
    num_robots = first_probs.shape[1]
    for robot in range(num_robots):
        for skill, label in enumerate(SKILL_LABELS):
            axes[1, 1].plot(
                iterations,
                first_probs[:, robot, skill],
                marker=("^", "s", "o")[skill],
                linestyle=("-", "--", "-.", ":")[robot % 4],
                label=f"r{robot} {label}",
            )
    axes[1, 1].set_title("First-step skill probabilities")
    axes[1, 1].legend(fontsize=7, ncol=2)
    timing = [
        float(_numpy(convergence[name]))
        for name in (
            "sampling_time_seconds",
            "rollout_time_seconds",
            "update_time_seconds",
        )
    ]
    axes[2, 0].bar(
        ["sampling", "rollout", "update"],
        timing,
        hatch=["/", "\\", "x"],
    )
    axes[2, 0].set_title("Planning time decomposition")
    axes[2, 1].axis("off")
    for axis in axes.flat:
        axis.grid(True, linestyle=":", alpha=0.25)
        if axis.has_data() and axis is not axes[2, 0]:
            axis.set_xlabel("CEM iteration")
    figure.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def plot_plan_value_diagnostics(rewards, values, uncertainty, gamma, output):
    """Plot diagnostic V(s_h) along the chosen path; only V(s_H) is optimized."""
    rewards = _numpy(rewards)
    values = _numpy(values)
    uncertainty = _numpy(uncertainty)
    discounted = rewards * np.power(float(gamma), np.arange(len(rewards)))
    cumulative = np.cumsum(discounted)
    figure, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(np.arange(1, len(rewards) + 1), rewards, "o-", label="predicted reward")
    axes[0].plot(np.arange(1, len(rewards) + 1), cumulative, "s--", label="discounted cumulative reward")
    axes[0].plot(np.arange(len(values)), values, "^-", label="diagnostic V(s_h)")
    axes[0].legend(); axes[0].grid(True, linestyle=":", alpha=.3)
    axes[1].plot(np.arange(1, len(uncertainty) + 1), uncertainty, "o-", color="tab:red")
    axes[1].set(xlabel="predicted macro step", ylabel="world-model uncertainty")
    axes[1].grid(True, linestyle=":", alpha=.3)
    figure.tight_layout(); output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160); plt.close(figure)


def plot_prediction_vs_reality(
    schema,
    predicted_states,
    actual_states,
    predicted_rewards,
    actual_rewards,
    uncertainty,
    event_probabilities=None,
    actual_events=None,
    event_names: Sequence[str] = (),
    output: Union[str, Path] = "prediction_vs_reality.png",
):
    predicted_states = _numpy(predicted_states)
    actual_states = _numpy(actual_states)
    predicted_rewards = _numpy(predicted_rewards)
    actual_rewards = _numpy(actual_rewards)
    uncertainty = _numpy(uncertainty)
    _, predicted_robot, predicted_ball = _field_coordinates(
        predicted_states, schema
    )
    _, actual_robot, actual_ball = _field_coordinates(actual_states, schema)
    error = np.square(predicted_states[1:] - actual_states[1:]).mean(-1)
    figure, axes = plt.subplots(3, 2, figsize=(13, 12))
    marker_choices = ("^", "s", "P", "X", "D", "v", "<", ">")
    for robot in range(predicted_robot.shape[-2]):
        marker = marker_choices[robot % len(marker_choices)]
        axes[0, 0].plot(
            predicted_robot[:, robot, 0],
            predicted_robot[:, robot, 1],
            linestyle="--",
            marker=marker,
            label=f"r{robot} predicted",
        )
        axes[0, 0].plot(
            actual_robot[:, robot, 0],
            actual_robot[:, robot, 1],
            linestyle="-",
            marker="P",
            label=f"r{robot} actual",
        )
    axes[0, 0].set_title("Robot position")
    axes[0, 1].plot(
        predicted_ball[:, 0],
        predicted_ball[:, 1],
        linestyle="--",
        marker="x",
        label="predicted",
    )
    axes[0, 1].plot(
        actual_ball[:, 0],
        actual_ball[:, 1],
        linestyle="-",
        marker="o",
        label="actual",
    )
    axes[0, 1].set_title("Ball position")
    axes[1, 0].plot(predicted_rewards, linestyle="--", marker="x", label="predicted")
    axes[1, 0].plot(actual_rewards, linestyle="-", marker="o", label="actual")
    axes[1, 0].set_title("Reward")
    axes[1, 1].plot(uncertainty, marker="s", label="uncertainty")
    axes[1, 1].plot(error, marker="x", linestyle="--", label="squared error")
    axes[1, 1].set_title("Uncertainty vs actual error")
    if event_probabilities is not None and actual_events is not None:
        probabilities = _numpy(event_probabilities)
        events = _numpy(actual_events)
        for index, name in enumerate(event_names):
            if events[:, index].any() or probabilities[:, index].max() > 0.25:
                axes[2, 0].plot(
                    probabilities[:, index], label=f"{name} probability"
                )
                axes[2, 0].step(
                    np.arange(len(events)),
                    events[:, index],
                    where="post",
                    linestyle="--",
                    label=f"{name} actual",
                )
        axes[2, 0].set_title("Predicted and real events")
    axes[2, 1].axis("off")
    for axis in axes.flat:
        if axis.has_data():
            axis.grid(True, linestyle=":", alpha=0.3)
            axis.legend(fontsize=7)
    figure.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)


@torch.no_grad()
def multi_step_prediction_errors(
    model,
    states,
    actions,
    rewards=None,
    max_horizon: Optional[int] = None,
):
    """Evaluate compounding model error under the actions actually executed.

    Unlike the tactical plan overlay, this keeps the future action sequence
    fixed to what happened in the simulator. It therefore isolates world-model
    rollout error from changes caused by receding-horizon replanning.
    """

    states = torch.as_tensor(states)
    actions = torch.as_tensor(actions)
    if states.ndim != 2 or actions.ndim != 2:
        raise ValueError("states and actions must be [time, feature] tensors")
    if states.shape[0] != actions.shape[0] + 1:
        raise ValueError("multi-step diagnostics require one more state than action")
    available = int(actions.shape[0])
    limit = min(
        available,
        int(max_horizon if max_horizon is not None else available),
    )
    if limit < 1:
        return []
    device = next(model.parameters()).device
    schema = model.schema
    reward_tensor = None if rewards is None else torch.as_tensor(rewards).float()
    metrics = []
    for horizon in range(1, limit + 1):
        starts = available - horizon + 1
        initial = states[:starts].to(device=device, dtype=torch.float)
        sequences = torch.stack(
            [actions[start : start + horizon] for start in range(starts)], dim=0
        ).to(device=device, dtype=torch.float)
        rollout = model.rollout(
            initial,
            sequences[:, None],
            deterministic=True,
            stop_on_done=False,
        )
        predicted = rollout["predicted_states"][:, 0, -1]
        actual = states[horizon : horizon + starts].to(
            device=device, dtype=predicted.dtype
        )
        scale = actual[:, schema.slice("field.geometry")][:, :2].abs()
        robot_errors = []
        for robot in range(model.action_adapter.num_robots):
            difference = (
                predicted[:, schema.slice(f"robot_{robot}.position")][:, :2]
                - actual[:, schema.slice(f"robot_{robot}.position")][:, :2]
            ) * scale
            robot_errors.append(difference.square().sum(dim=-1))
        robot_position_rmse_m = torch.stack(robot_errors, dim=-1).mean().sqrt()
        ball_difference = (
            predicted[:, schema.slice("ball.position")][:, :2]
            - actual[:, schema.slice("ball.position")][:, :2]
        ) * scale
        ball_position_rmse_m = ball_difference.square().sum(dim=-1).mean().sqrt()
        state_rmse = (predicted - actual).square().mean().sqrt()
        row = {
            "horizon": horizon,
            "samples": starts,
            "state_rmse": float(state_rmse.detach().cpu()),
            "robot_position_rmse_m": float(
                robot_position_rmse_m.detach().cpu()
            ),
            "ball_position_rmse_m": float(ball_position_rmse_m.detach().cpu()),
        }
        if reward_tensor is not None:
            actual_returns = torch.stack(
                [
                    reward_tensor[start : start + horizon].sum()
                    for start in range(starts)
                ]
            ).to(device=device, dtype=predicted.dtype)
            predicted_returns = rollout["predicted_rewards"][:, 0].sum(dim=-1)
            row["cumulative_reward_mae"] = float(
                (predicted_returns - actual_returns).abs().mean().detach().cpu()
            )
        metrics.append(row)
    return metrics


def plot_mpc_execution_diagnostics(
    step_diagnostics,
    multi_step_errors,
    output: Union[str, Path] = "mpc_diagnostics.png",
):
    """Plot fallback/action modification and compounding model errors."""

    steps = np.arange(len(step_diagnostics))
    fallback = np.asarray([row["fallback_used"] for row in step_diagnostics])
    modified = np.asarray(
        [row["requested_action_modified"] for row in step_diagnostics]
    )
    objective = np.asarray([row["best_objective"] for row in step_diagnostics])
    planning = np.asarray(
        [row["planning_time_seconds"] for row in step_diagnostics]
    )
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes[0, 0].step(steps, fallback, where="post", label="fallback used")
    axes[0, 0].step(
        steps, modified, where="post", linestyle="--", label="action modified"
    )
    axes[0, 0].set(ylim=(-0.1, 1.1), title="Execution interventions")
    axes[0, 0].legend()
    axes[0, 1].plot(steps, objective, marker="o", label="best objective")
    axes[0, 1].set_title("Selected-plan objective")
    axes[1, 0].plot(steps, planning, marker="s", color="tab:purple")
    axes[1, 0].set(title="Planning latency", ylabel="seconds")
    horizons = np.asarray([row["horizon"] for row in multi_step_errors])
    if len(horizons):
        axes[1, 1].plot(
            horizons,
            [row["robot_position_rmse_m"] for row in multi_step_errors],
            "o-",
            label="robot position RMSE",
        )
        axes[1, 1].plot(
            horizons,
            [row["ball_position_rmse_m"] for row in multi_step_errors],
            "s-",
            label="ball position RMSE",
        )
    axes[1, 1].set(
        title="World-model rollout error under executed actions",
        xlabel="macro-step horizon",
        ylabel="metres",
    )
    axes[1, 1].legend()
    for axis in axes.flat:
        axis.grid(True, linestyle=":", alpha=0.3)
        if axis is not axes[1, 1]:
            axis.set_xlabel("executed macro step")
    figure.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


def plot_skill_and_parameters(
    actions,
    num_robots: int,
    output: Union[str, Path] = "skill_and_parameters.png",
):
    """Plot the executed MPC skill and command parameters for each robot."""

    actions = _numpy(actions)
    expected_width = 4 * int(num_robots)
    if actions.ndim != 2 or actions.shape[1] != expected_width:
        raise ValueError(
            f"Expected actions [steps, {expected_width}], got {actions.shape}"
        )
    actions = actions.reshape(actions.shape[0], int(num_robots), 4)
    skills = actions[..., 0].astype(int)
    parameters = actions[..., 1:]
    steps = np.arange(actions.shape[0])
    colors = [plt.get_cmap("tab10")(robot % 10) for robot in range(num_robots)]

    figure, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    for robot in range(num_robots):
        axes[0].step(
            steps,
            skills[:, robot],
            where="post",
            color=colors[robot],
            linewidth=2,
            label=f"robot {robot}",
        )
    axes[0].set_yticks(range(len(SKILL_LABELS)), SKILL_LABELS)
    axes[0].set_ylim(-0.35, len(SKILL_LABELS) - 0.65)
    axes[0].set_ylabel("selected skill")
    axes[0].legend(ncol=max(1, min(num_robots, 4)), fontsize=8)

    parameter_labels = (
        "parameter 0: x (walk: body-forward; ball skills: field-x)",
        "parameter 1: y (walk: body-left; ball skills: field-y)",
        "parameter 2: yaw-rate command",
    )
    for parameter, label in enumerate(parameter_labels):
        axis = axes[parameter + 1]
        for robot in range(num_robots):
            axis.step(
                steps,
                parameters[:, robot, parameter],
                where="post",
                color=colors[robot],
                linewidth=1.6,
                label=f"robot {robot}",
            )
        axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
        axis.set_ylabel(label)

    axes[-1].set_xlabel("executed MPC macro step")
    for axis in axes:
        axis.grid(True, linestyle=":", alpha=0.3)
    figure.suptitle(
        "Executed MPC skills and parameters\n"
        "(parameter 2 is unused and zero for shoot)",
        fontsize=12,
    )
    figure.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


def figure_to_rgb(figure) -> np.ndarray:
    figure.canvas.draw()
    rgba = np.asarray(figure.canvas.buffer_rgba())
    return rgba[..., :3].copy()


def save_video_or_frames(
    frames: Sequence[np.ndarray],
    output: Union[str, Path],
    fps: float,
) -> Path:
    """Save an MP4 directly without writing a persistent frame directory."""

    import imageio.v2 as imageio

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        imageio.mimsave(output, list(frames), fps=fps)
        return output
    except Exception as exc:
        raise RuntimeError(
            "Could not encode the requested MPC MP4. Install an imageio "
            "FFmpeg backend; no frame directory or fallback files were written."
        ) from exc
