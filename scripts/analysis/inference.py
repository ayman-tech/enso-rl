"""
Unified inference pass: paired rollouts → raw per-step data for lift + seasonality.

Runs `n_rollouts` paired (agent vs baseline, same random seed) rollouts for each
ensemble seed. Saves the full per-step observations, actions, and MYE labels to a
single rich npz so the notebook can do all aggregation, statistics, and plotting
without re-running simulation.

Output: plots/{model}/inference.npz

npz arrays:
  var_names           [10]                      XRO mode names (Nino3.4 … SASD)
  action_names        [9]                       forcing mode names (var_names[1:])
  seeds               [n_seeds]                 model seed indices used
  rollout_seeds       [n_seeds, n_rollouts]      RNG seed for each rollout pair
  rollout_start_month [n_seeds, n_rollouts]      0-based calendar month of step 0
                                                 (reset() starts the seasonal clock at the
                                                 sampled state's month; needed to bin by
                                                 TRUE calendar month = (step+start)%12)
  n_rollouts          scalar
  months              scalar                    T (steps per rollout)
  spinup              scalar (=12)              months to skip for seasonality binning

  agent_obs           [n_seeds, n_rollouts, T, 10]  full state after each step
  agent_actions       [n_seeds, n_rollouts, T, 9]   scaled forcing applied
  agent_mye_label     [n_seeds, n_rollouts, T]       int8 MYE category (see below)

  base_obs            [n_seeds, n_rollouts, T, 10]  baseline (no agent) state
  base_mye_label      [n_seeds, n_rollouts, T]

MYE label encoding:
  0 = Neutral
  1 = Single-year El Niño
  2 = Single-year La Niña
  3 = Multi-year El Niño
  4 = Multi-year La Niña

Notebook recipes:
  nino34_agent  = d['agent_obs'][:, :, :, 0]          # Nino3.4 time series
  tna_agent     = d['agent_obs'][:, :, :, var_names.index('TNA')]
  mye_mask      = d['agent_mye_label'] >= 3            # any MYE month
  lift_per_roll = mye_mask.mean(-1) - (d['base_mye_label'] >= 3).mean(-1)

Alignment note:
  states_traj from simulate_trajectory is shape (T+1, 10) — initial obs prepended.
  We store states_traj[1:] so index t aligns with action t and mye_label t.

Usage:
    uv run scripts/analysis/inference.py --model ensemble
    uv run scripts/analysis/inference.py --model ensemble --n-rollouts 5 --months 120

Seeds are auto-detected from all existing models/{model}_seed*.zip files.
"""
import re
import sys
import time
import argparse
import numpy as np
import multiprocessing as mp
from multiprocessing import cpu_count
from pathlib import Path

repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from utils.model_io import load_environment
from utils.evaluation import simulate_trajectory
from utils import suppress_warnings
from config import EnvConfig

SPINUP = 12  # months to skip as transient at the start of each rollout. NOTE: reset()
             # starts the seasonal clock at the sampled state's calendar month (not Jan),
             # so bin by TRUE calendar month via rollout_start_month, not step % 12.

# String labels produced by classify_enso_event → int8 encoding
_LABEL_MAP = {
    'Neutral':              0,
    'Single-year El Nino':  1,
    'Single-year La Nina':  2,
    'Multi-year El Nino':   3,
    'Multi-year La Nina':   4,
}


def _encode_classified(classified_array):
    """Convert string classified_event_array [T+1] → int8 [T] (skip index 0)."""
    return np.array([_LABEL_MAP.get(s, 0) for s in classified_array[1:]], dtype=np.int8)


def _seed_sort_key(s):
    """Sort ints numerically first, then any string suffixes lexicographically."""
    return (0, s, "") if isinstance(s, int) else (1, 0, str(s))


def discover_seeds(model):
    """Find ensemble seeds from existing models/{model}_seed*.zip filenames.

    The suffix after `_seed` is returned as an int when it is a plain integer
    (the ensemble case, e.g. ensemble_seed3.zip), otherwise as the raw string
    (e.g. a sweep model's 0-0-0-0-3). Returns a sorted list.
    """
    seeds = []
    pat = re.compile(rf"{re.escape(model)}_seed(.+)")
    for p in sorted(Path("models").glob(f"{model}_seed*.zip")):
        m = pat.fullmatch(p.stem)
        if not m:
            continue
        tok = m.group(1)
        seeds.append(int(tok) if tok.lstrip("-").isdigit() else tok)
    return sorted(seeds, key=_seed_sort_key)


def _print_summary(seeds_used, agent_mye_label, base_mye_label):
    print(f"\n{'='*55}")
    print(f"{'Seed':>6} | {'Agent MYE%':>10} | {'Base MYE%':>10} | {'Lift':>8}")
    print(f"{'-'*55}")
    for si, seed in enumerate(seeds_used):
        a_frac = (agent_mye_label[si] >= 3).mean() * 100
        b_frac = (base_mye_label[si] >= 3).mean() * 100
        print(f"{seed:>6} | {a_frac:>10.1f}% | {b_frac:>10.1f}% | {a_frac - b_frac:>+7.1f}%")
    overall_a = (agent_mye_label >= 3).mean() * 100
    overall_b = (base_mye_label >= 3).mean() * 100
    print(f"{'all':>6} | {overall_a:>10.1f}% | {overall_b:>10.1f}% | {overall_a - overall_b:>+7.1f}%")
    print(f"{'='*55}")


def _worker_seed(wargs):
    """Run all paired rollouts for one trained seed.

    Spawn-safe: heavy imports and model loading happen here, not in the parent.
    The rollout seeds are precomputed by the parent and passed in, so results are
    bit-identical to the serial version and independent of completion order.
    Returns the per-seed arrays for the parent to assemble in seed order.
    """
    si, seed, model_name, r_seeds, T, env_config = wargs
    suppress_warnings()
    model, env, vnames = load_environment(model_name, env_config)

    n_r = len(r_seeds)
    a_obs = np.zeros((n_r, T, 10), dtype=np.float32)
    a_act = np.zeros((n_r, T,  9), dtype=np.float32)
    a_mye = np.zeros((n_r, T),     dtype=np.int8)
    b_obs = np.zeros((n_r, T, 10), dtype=np.float32)
    b_mye = np.zeros((n_r, T),     dtype=np.int8)
    a_start = np.zeros(n_r,        dtype=np.int8)  # 0-based calendar month of step 0

    for ri, rs in enumerate(r_seeds):
        sim_a = simulate_trajectory(env, agent=model, num_months=T, seed=rs)
        sim_b = simulate_trajectory(env, agent=None,  num_months=T, seed=rs)
        a_obs[ri] = sim_a['states_traj'][1:].astype(np.float32)
        a_act[ri] = sim_a['actions_traj'].astype(np.float32)
        a_mye[ri] = _encode_classified(sim_a['classified_event_array'])
        b_obs[ri] = sim_b['states_traj'][1:].astype(np.float32)
        b_mye[ri] = _encode_classified(sim_b['classified_event_array'])
        # agent & baseline share the same seed -> same start month; record once.
        a_start[ri] = sim_a['month_offset']

    af = (a_mye >= 3).mean() * 100
    bf = (b_mye >= 3).mean() * 100
    print(f"  [seed {seed}] done — agent={af:.1f}%  base={bf:.1f}%  lift={af-bf:+.1f}%", flush=True)
    return si, seed, vnames, a_obs, a_act, a_mye, b_obs, b_mye, a_start


def main():
    parser = argparse.ArgumentParser(description="Unified inference: paired rollouts → raw npz")
    parser.add_argument("--model",       type=str, default="ensemble",
                        help="Ensemble prefix — auto-loads all models/{model}_seed*.zip")
    parser.add_argument("--n-rollouts",  type=int, default=30,
                        help="Paired rollouts per seed")
    parser.add_argument("--months",      type=int, default=1200,
                        help="Simulation months per rollout (T)")
    parser.add_argument("--master-seed", type=int, default=42)
    parser.add_argument("--workers",     type=int, default=None,
                        help="Parallel seeds at once (default: cpu_count - 2; use 1 "
                             "for serial). Mirrors shapley/counterfactual_analysis; "
                             "results are identical to serial.")
    args = parser.parse_args()

    suppress_warnings()
    start_time = time.time()

    # Auto-detect all trained seeds from the model files.
    valid_seeds = discover_seeds(args.model)
    if not valid_seeds:
        raise FileNotFoundError(
            f"No models/{args.model}_seed*.zip found — check --model.")

    output_dir = Path("plots") / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "inference.npz"

    env_config = EnvConfig()
    master_rng = np.random.default_rng(args.master_seed)

    print("=" * 60)
    print("INFERENCE — paired rollouts → raw per-step data")
    print("=" * 60)
    print(f"  model      = {args.model}")
    print(f"  seeds      = {valid_seeds}")
    print(f"  n_rollouts = {args.n_rollouts}")
    print(f"  months     = {args.months}")
    print(f"  output     = {out_path}")
    print()

    n_seeds = len(valid_seeds)
    T = args.months
    n_r = args.n_rollouts

    # Pre-allocate
    agent_obs       = np.zeros((n_seeds, n_r, T, 10), dtype=np.float32)
    agent_actions   = np.zeros((n_seeds, n_r, T,  9), dtype=np.float32)
    agent_mye_label = np.zeros((n_seeds, n_r, T),     dtype=np.int8)
    base_obs        = np.zeros((n_seeds, n_r, T, 10), dtype=np.float32)
    base_mye_label  = np.zeros((n_seeds, n_r, T),     dtype=np.int8)
    rollout_seeds   = np.zeros((n_seeds, n_r),         dtype=np.int64)
    rollout_start_month = np.zeros((n_seeds, n_r),     dtype=np.int8)  # 0-based calendar month of step 0

    # Common random numbers ACROSS trained seeds: derive ONE set of rollout seeds
    # from the master RNG and score every model on it. Because each model sees the
    # identical eval draws, differencing across models cancels evaluation-draw noise,
    # so the across-seed spread reflects only the trained weights — not which eval
    # conditions a given model happened to be tested on. (Depends only on
    # --master-seed, so it stays reproducible and order-independent for parallelism.)
    r_seeds = [int(master_rng.integers(0, 2**31)) for _ in range(n_r)]
    worker_args = []
    for si, seed in enumerate(valid_seeds):
        rollout_seeds[si] = r_seeds
        worker_args.append((si, seed, f"{args.model}_seed{seed}", r_seeds, T, env_config))

    n_workers = args.workers if args.workers else max(1, cpu_count() - 2)
    n_workers = min(n_workers, n_seeds)  # no more workers than seeds

    if n_workers > 1:
        print(f"  Parallel across {n_seeds} seeds with {n_workers} workers\n", flush=True)
        with mp.Pool(processes=n_workers) as pool:
            results = pool.map(_worker_seed, worker_args)
    else:
        print("  Serial (workers=1)\n", flush=True)
        results = [_worker_seed(wa) for wa in worker_args]

    # Assemble per-seed results into the pre-allocated arrays (by seed index).
    var_names = None
    for si, seed, vnames, a_obs, a_act, a_mye, b_obs, b_mye, a_start in results:
        if var_names is None:
            var_names = vnames
        agent_obs[si]       = a_obs
        agent_actions[si]   = a_act
        agent_mye_label[si] = a_mye
        base_obs[si]        = b_obs
        base_mye_label[si]  = b_mye
        rollout_start_month[si] = a_start

    _print_summary(valid_seeds, agent_mye_label, base_mye_label)

    np.savez(
        out_path,
        var_names=np.array(var_names),
        action_names=np.array(list(var_names[1:])),
        seeds=np.array(valid_seeds),
        rollout_seeds=rollout_seeds,
        rollout_start_month=rollout_start_month,
        n_rollouts=n_r,
        months=T,
        spinup=SPINUP,
        agent_obs=agent_obs,
        agent_actions=agent_actions,
        agent_mye_label=agent_mye_label,
        base_obs=base_obs,
        base_mye_label=base_mye_label,
    )
    print(f"\nSaved → {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")

    elapsed = time.time() - start_time
    h, rem = divmod(int(elapsed), 3600)
    m, s = divmod(rem, 60)
    print(f"\n{'='*60}")
    print(f"INFERENCE COMPLETE — {h}h {m}m {s}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
