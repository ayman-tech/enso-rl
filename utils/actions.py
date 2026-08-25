"""Single source of truth for month-indexed action scaling.

XROMultiYearEnv.reset() starts the seasonal clock at the SAMPLED state's real
calendar month (month_offset = sampled_month - 1), so the scale the physics
applies at step t is column

    (t + start_month) % 12        NOT   t % 12

Every producer and consumer of scaled actions must go through this module. Two
independent recorders (utils/evaluation.py and scripts/analysis/mutual_information.py)
previously hand-rolled `step % 12` and silently disagreed with the physics for any
rollout that did not start in January -- which is ~90% of them.
"""
import numpy as np

N_MONTHS = 12
N_MODES = 9


def default_action_scale():
    """[9, 12] scale from EnvConfig (already halved in EnvConfig.__post_init__).

    Imported lazily so utils.physics stays free of any config dependency.
    """
    from config import EnvConfig
    return np.asarray(EnvConfig().action_scale, dtype=np.float64)


def _as_matrix(action_scale):
    """Validate/normalize an action-scale matrix to [9, 12] float64."""
    m = default_action_scale() if action_scale is None else np.asarray(action_scale, dtype=np.float64)
    if m.shape != (N_MODES, N_MONTHS):
        raise ValueError(
            f"action_scale must have shape ({N_MODES}, {N_MONTHS}) "
            f"(modes x calendar months), got {m.shape}"
        )
    return m


def calendar_month(step, start_month=0):
    """0-based TRUE calendar month at `step`. Broadcasts over both arguments."""
    return (np.asarray(step) + np.asarray(start_month)) % N_MONTHS


def scale_for_step(step, start_month=0, action_scale=None):
    """[9] scale applied at a single step. Used by xro_step."""
    return _as_matrix(action_scale)[:, int(calendar_month(step, start_month))]


def monthly_scale(n_steps, start_month=0, action_scale=None):
    """[..., T, 9] scale per step.

    `start_month` may be a scalar or any shape (e.g. [n_seeds, n_rollouts]); its
    leading dimensions are preserved in the result.
    """
    m = _as_matrix(action_scale)
    cal = calendar_month(np.arange(n_steps), np.asarray(start_month)[..., None])
    return np.moveaxis(m[:, cal], 0, -1)


def scale_actions(raw, start_month=0, action_scale=None):
    """raw [..., T, 9] in [-1, 1] -> applied forcing in each mode's physical units."""
    raw = np.asarray(raw, dtype=np.float64)
    return raw * monthly_scale(raw.shape[-2], start_month, action_scale)


def unscale_actions(scaled, start_month=0, action_scale=None):
    """Inverse of scale_actions: applied forcing -> raw policy action in [-1, 1]."""
    scaled = np.asarray(scaled, dtype=np.float64)
    return scaled / monthly_scale(scaled.shape[-2], start_month, action_scale)
