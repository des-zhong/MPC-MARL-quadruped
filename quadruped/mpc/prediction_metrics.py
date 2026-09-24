"""Physical one-step diagnostics shared by online world-model training."""


def robot_position_errors(schema, prediction, target, live):
    """Mean planar Euclidean error (meters) per robot and across robots.

    XY positions are normalized by field geometry in the encoded state.
    Exclude terminal/reset transitions using the caller's complete-transition
    mask. Robot indices include both teams in world-model schema order.
    Returns no measurements when the batch has no complete transitions.
    """
    if not live.any():
        return {}
    scale = target[live, schema.slice('field.geometry')][:, :2].abs()
    metrics = {}
    for feature in schema.features:
        if not (feature.name.startswith('robot_') and feature.name.endswith('.position')):
            continue
        position = schema.slice(feature.name)
        delta = (prediction[live, position][:, :2] - target[live, position][:, :2]) * scale
        robot = feature.name.split('.')[0]
        metrics[f'mpc_quality/{robot}_position_error_m'] = float(delta.norm(dim=-1).mean())
    if metrics:
        metrics['mpc_quality/robot_position_error_m'] = sum(metrics.values()) / len(metrics)
    return metrics
