"""
Per-axis random-seed resolution for the randomness-sensitivity study.

A training run draws randomness from five independent axes:

    1. weight  - PPO policy weight initialization (PyTorch RNG)
    2. action  - stochastic Gaussian exploration / action sampling (PyTorch RNG)
    3. shuffle - PPO mini-batch shuffle each epoch (global NumPy RNG)
    4. init    - environment start-state sampling (env's rng_init)
    5. physics - XRO climate noise each step (env's rng_physics)

By default every axis takes the master `--seed` directly, so `--seed 5` makes all
five axes use 5 (a single reproducible run / ensemble member). To study sensitivity
to one axis, override it (e.g. `--seed 5 --seed-physics 3`) so that axis varies while
the other four stay pinned at the master — the standard "common random numbers"
attribution design. No SeedSequence spawning: each axis seed is just the override or
the master, which keeps control direct and filenames legible.
"""
from dataclasses import dataclass
from typing import Optional, Dict

# Fixed axis order used for the bundle tuple and the model-name suffix.
AXES = ("weight", "action", "shuffle", "init", "physics")


@dataclass
class SeedBundle:
    """Resolved seed for each randomness axis (None = unseeded)."""
    weight: Optional[int] = None
    action: Optional[int] = None
    shuffle: Optional[int] = None
    init: Optional[int] = None
    physics: Optional[int] = None

    @property
    def has_override(self) -> bool:
        """True if any axis differs from the others (i.e. an override was applied)."""
        vals = (self.weight, self.action, self.shuffle, self.init, self.physics)
        return len(set(vals)) > 1

    def as_tuple(self):
        return (self.weight, self.action, self.shuffle, self.init, self.physics)

    def as_log_dict(self) -> Dict[str, Optional[int]]:
        """Flat dict for W&B config / stdout provenance."""
        return {f"seed_{ax}": getattr(self, ax) for ax in AXES}


def resolve_seeds(master: Optional[int], overrides: Dict[str, Optional[int]]) -> SeedBundle:
    """Resolve the five axis seeds.

    Each axis = its override if provided, else the master seed.

    Args:
        master: Master seed (`--seed`). None means an unseeded run.
        overrides: Mapping of axis name -> override int (or None). Keys must be in AXES.

    Returns:
        SeedBundle with every axis populated.

    Raises:
        ValueError: if an override is given without a master seed (the un-overridden
            axes would otherwise be None, producing meaningless mixed/None runs).
    """
    overrides = overrides or {}
    bad = set(overrides) - set(AXES)
    if bad:
        raise ValueError(f"Unknown seed axis override(s): {sorted(bad)}; valid: {AXES}")

    any_override = any(v is not None for v in overrides.values())
    if any_override and master is None:
        raise ValueError(
            "A per-axis seed override was given without --seed. Pass --seed <master> so "
            "the non-overridden axes are pinned (otherwise they would be unseeded)."
        )

    resolved = {ax: (overrides.get(ax) if overrides.get(ax) is not None else master)
                for ax in AXES}
    return SeedBundle(**resolved)


def model_name(base: str, bundle: SeedBundle) -> str:
    """Build the model save name.

    With any override, encode all five effective seeds so ablation runs are
    distinguishable on disk: `<base>_seed<weight>-<action>-<shuffle>-<init>-<physics>`.
    Without an override, return `base` unchanged (the master-only naming convention,
    e.g. train_ensemble.py's `<prefix>_seed<master>`, stays the caller's job).
    """
    if not bundle.has_override:
        return base
    w, a, b, i, p = bundle.as_tuple()
    return f"{base}_seed{w}-{a}-{b}-{i}-{p}"
