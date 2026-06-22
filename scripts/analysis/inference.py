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
    uv run scripts/analysis/inference.py --model ensemble --seeds 0 1 2 --n-rollouts 5 --months 120
"""
import sys
import time
import argparse
import numpy as np
from pathlib import Path

repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from utils.model_io import load_environment
from utils.evaluation import simulate_trajectory
from utils import suppress_warnings
from config import EnvConfig

SPINUP = 12  # fixed by XRO env: seasonal clock resets to Jan on every env.reset()

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


def main():
    parser = argparse.ArgumentParser(description="Unified inference: paired rollouts → raw npz")
    parser.add_argument("--model",       type=str, default="ensemble",
                        help="Ensemble prefix — loads models/{model}_seed{s}.zip")
    parser.add_argument("--seeds",       type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--n-rollouts",  type=int, default=30,
                        help="Paired rollouts per seed")
    parser.add_argument("--months",      type=int, default=1200,
                        help="Simulation months per rollout (T)")
    parser.add_argument("--master-seed", type=int, default=42)
    args = parser.parse_args()

    suppress_warnings()
    start_time = time.time()

    output_dir = Path("plots") / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "inference.npz"

    env_config = EnvConfig()
    master_rng = np.random.default_rng(args.master_seed)

    print("=" * 60)
    print("INFERENCE — paired rollouts → raw per-step data")
    print("=" * 60)
    print(f"  model      = {args.model}")
    print(f"  seeds      = {args.seeds}")
    print(f"  n_rollouts = {args.n_rollouts}")
    print(f"  months     = {args.months}")
    print(f"  output     = {out_path}")
    print()

    # Discover how many seeds actually have models
    valid_seeds = []
    for s in args.seeds:
        p = Path(f"models/{args.model}_seed{s}.zip")
        if p.exists():
            valid_seeds.append(s)
        else:
            print(f"  [skip] models/{args.model}_seed{s}.zip not found")

    if not valid_seeds:
        raise FileNotFoundError("No model files found. Check --model and --seeds.")

    n_seeds = len(valid_seeds)
    T = args.months
    n_r = args.n_rollouts
    var_names = None

    # Pre-allocate
    agent_obs       = np.zeros((n_seeds, n_r, T, 10), dtype=np.float32)
    agent_actions   = np.zeros((n_seeds, n_r, T,  9), dtype=np.float32)
    agent_mye_label = np.zeros((n_seeds, n_r, T),     dtype=np.int8)
    base_obs        = np.zeros((n_seeds, n_r, T, 10), dtype=np.float32)
    base_mye_label  = np.zeros((n_seeds, n_r, T),     dtype=np.int8)
    rollout_seeds   = np.zeros((n_seeds, n_r),         dtype=np.int64)

    for si, seed in enumerate(valid_seeds):
        name = f"{args.model}_seed{seed}"
        print(f"\n── Seed {seed}  ({name}) ──")
        model, env, vnames = load_environment(name, env_config)
        if var_names is None:
            var_names = vnames

        seed_rng = np.random.default_rng(master_rng.integers(0, 2**31))
        r_seeds = [int(seed_rng.integers(0, 2**31)) for _ in range(n_r)]

        for ri, rs in enumerate(r_seeds):
            sim_a = simulate_trajectory(env, agent=model, num_months=T, seed=rs)
            sim_b = simulate_trajectory(env, agent=None,  num_months=T, seed=rs)

            agent_obs[si, ri]       = sim_a['states_traj'][1:].astype(np.float32)
            agent_actions[si, ri]   = sim_a['actions_traj'].astype(np.float32)
            agent_mye_label[si, ri] = _encode_classified(sim_a['classified_event_array'])
            base_obs[si, ri]        = sim_b['states_traj'][1:].astype(np.float32)
            base_mye_label[si, ri]  = _encode_classified(sim_b['classified_event_array'])
            rollout_seeds[si, ri]   = rs

            if (ri + 1) % 5 == 0 or ri == n_r - 1:
                a_mye = (agent_mye_label[si, :ri + 1] >= 3).mean() * 100
                b_mye = (base_mye_label[si, :ri + 1] >= 3).mean() * 100
                print(f"  rollout {ri+1:>3}/{n_r}  agent={a_mye:.1f}%  base={b_mye:.1f}%")

    _print_summary(valid_seeds, agent_mye_label, base_mye_label)

    np.savez(
        out_path,
        var_names=np.array(var_names),
        action_names=np.array(list(var_names[1:])),
        seeds=np.array(valid_seeds),
        rollout_seeds=rollout_seeds,
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
