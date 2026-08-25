"""
Mutual Information Analysis between RL actions and ENSO outcomes.

Computes MI(action_i, future_ENSO_classification) to identify which action
dimensions carry the most information about future ENSO state.

Uses N independent paired-seed runs for statistical robustness with t-tests,
95% CIs, and significance testing. Each run produces an independent MI estimate
from a separate trajectory.

Methods:
  - Histogram-based MI estimation (per trajectory)
  - Permutation-based significance testing
  - Time-lagged MI profiles
  - Statistical analysis across independent runs

Usage:
    python scripts/mutual_information.py --model rl_model
    python scripts/mutual_information.py --model rl_model --n-runs 30 --months 6000
"""
import sys
import time
import argparse
import numpy as np
from pathlib import Path
from scipy import stats as sp_stats

# scripts/analysis/mutual_information.py -> three levels up is the repo root
# (was two, which pointed at scripts/ and made `import config` fail).
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from stable_baselines3 import PPO
from config import EnvConfig
from utils.data_processing import load_observational_data, prepare_xro_parameters
from envs import XROMultiYearEnv
from utils import suppress_warnings
from utils.evaluation import simulate_trajectory
from XRO.core import XRO


def load_environment(model_path: str, env_config: EnvConfig):
    """Load trained model and create environment."""
    model_path_str = model_path
    if not model_path_str.endswith('.zip'):
        model_path_str += '.zip'
    if not model_path_str.startswith('models'):
        model_path_str = f'models/{model_path_str}'

    model_path = Path(model_path_str)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    obs_ds, train_ds, var_names, bounds = load_observational_data(
        env_config.data_config["data_path"],
        env_config.data_config["train_start"],
        env_config.data_config["train_end"]
    )

    model_xro = XRO()
    params = prepare_xro_parameters(model_xro, train_ds, var_names,
                                    config=env_config)
    params['threshold'] = env_config.threshold

    env = XROMultiYearEnv(
        params=params, train_ds=train_ds,
        var_names=var_names, max_steps=env_config.max_steps
    )

    model = PPO.load(str(model_path), env=env)
    return model, env, var_names


def collect_single_trajectory(env, model, num_months, seed):
    """
    Collect actions and Nino3.4 values from a single trajectory.

    Delegates to the shared recorder so the month-indexed action scaling has exactly
    one implementation (utils.actions). The hand-rolled loop this replaced carried two
    bugs:
      1. it scaled actions by `step % 12`, ignoring env.month_offset -- wrong for the
         ~90% of rollouts that do not start in January;
      2. it read Nino3.4 from `obs_next[0]`, which is the Z-SCORED observation
         (XROMultiYearEnv._get_obs), and then classified it against a threshold in raw
         degC. With the ORAS5 climatology (mean -0.0232, std 0.8887) that made the
         effective thresholds +0.421 / -0.468 degC -- asymmetric, and not +/-0.5.
         states_traj carries env.state, which is raw.

    Args:
        env: Environment
        model: Trained PPO model
        num_months: Months to simulate
        seed: Random seed for env.reset()

    Returns:
        actions: [num_months, 9] RAW policy actions in [-1, 1]
        nino34:  [num_months] raw Nino3.4 (degC)
    """
    sim = simulate_trajectory(env, agent=model, num_months=num_months, seed=seed)
    # states_traj is [T+1, 10] with the initial state prepended, so [1:] aligns index
    # t with action t (same convention as scripts/analysis/inference.py).
    return sim['raw_actions_traj'], sim['states_traj'][1:, 0]


def classify_nino34(nino34, threshold=0.5):
    """
    Classify Nino3.4 into ENSO categories.

    Returns:
      0 = La Nina (< -threshold)
      1 = Neutral
      2 = El Nino (> threshold)
    """
    classes = np.ones(len(nino34), dtype=int)
    classes[nino34 > threshold] = 2
    classes[nino34 < -threshold] = 0
    return classes


def histogram_mutual_information(x, y_discrete, n_bins=20):
    """
    Estimate MI(X; Y) where X is continuous and Y is discrete.

    Uses binned histogram approach:
      MI = sum_y p(y) * sum_x p(x|y) * log(p(x|y) / p(x))

    Args:
        x: Continuous variable [N]
        y_discrete: Discrete class labels [N]
        n_bins: Number of bins for X

    Returns:
        float: MI estimate in nats
    """
    classes = np.unique(y_discrete)

    x_min, x_max = x.min() - 1e-10, x.max() + 1e-10
    bins = np.linspace(x_min, x_max, n_bins + 1)
    x_binned = np.digitize(x, bins) - 1
    x_binned = np.clip(x_binned, 0, n_bins - 1)

    n = len(x)
    px = np.zeros(n_bins)
    for b in range(n_bins):
        px[b] = np.sum(x_binned == b) / n
    px = np.maximum(px, 1e-12)

    mi = 0.0
    for c in classes:
        mask = (y_discrete == c)
        py_c = np.sum(mask) / n
        if py_c < 1e-12:
            continue

        px_given_y = np.zeros(n_bins)
        for b in range(n_bins):
            px_given_y[b] = np.sum((x_binned == b) & mask) / np.sum(mask)
        px_given_y = np.maximum(px_given_y, 1e-12)

        for b in range(n_bins):
            if px_given_y[b] > 1e-10 and px[b] > 1e-10:
                mi += py_c * px_given_y[b] * np.log(px_given_y[b] / px[b])

    return max(mi, 0.0)


def compute_mi_for_trajectory(actions, nino34, lag, threshold=0.5, n_bins=20):
    """
    Compute MI between each action dimension and ENSO classification at a future lag.

    MI(action_i(t), ENSO_class(t + lag))

    Args:
        actions: [T, 9] action array
        nino34: [T] Nino3.4 array
        lag: Lag in months
        threshold: ENSO classification threshold
        n_bins: Histogram bins

    Returns:
        np.ndarray: MI values [9]
    """
    T = len(actions)
    if lag >= T:
        return np.zeros(actions.shape[1])

    actions_aligned = actions[:T - lag]
    nino34_future = nino34[lag:]
    enso_class = classify_nino34(nino34_future, threshold)

    mi_values = np.zeros(actions.shape[1])
    for i in range(actions.shape[1]):
        mi_values[i] = histogram_mutual_information(actions_aligned[:, i], enso_class, n_bins)

    return mi_values


def compute_mi_for_run(env, model, num_months, seed, lags, threshold, n_bins):
    """
    Compute MI at multiple lags for a single independent run.

    MI is computed on RAW actions. MI is invariant to a *fixed* monotone transform,
    but the per-month action scale is month-DEPENDENT, so pooling 1200 months of
    scaled forcing mixes 12 different stretches into one set of histogram bins and
    MI(scaled) != MI(raw). Raw is also unit-free, so the 9 modes -- whose scales
    differ by ~10x -- are directly comparable.

    Args:
        env: Environment
        model: Trained model
        num_months: Months per trajectory
        seed: Random seed
        lags: List of lag values
        threshold: ENSO classification threshold
        n_bins: Histogram bins

    Returns:
        lag_profiles: [n_lags, 9] MI values at each lag
        actions: [T, 9] raw trajectory actions in [-1, 1]
        nino34: [T] trajectory Nino3.4 (raw degC)
    """
    actions, nino34 = collect_single_trajectory(env, model, num_months, seed)

    lag_profiles = np.zeros((len(lags), actions.shape[1]))
    for i, lag in enumerate(lags):
        lag_profiles[i] = compute_mi_for_trajectory(actions, nino34, lag, threshold, n_bins)

    return lag_profiles, actions, nino34


def compute_statistics(values_per_run, feature_names):
    """
    Compute paired-run statistics on MI values.

    Args:
        values_per_run: [N_RUNS, n_features] MI values per run
        feature_names: list of action names

    Returns:
        list of dicts with per-feature statistics
    """
    n_runs = values_per_run.shape[0]
    stats = []

    for i, name in enumerate(feature_names):
        vals = values_per_run[:, i]
        mean_val = vals.mean()
        std_val = vals.std(ddof=1)
        se_val = std_val / np.sqrt(n_runs)
        ci_95 = sp_stats.t.ppf(0.975, df=n_runs - 1) * se_val

        # t-test: is MI significantly > 0?
        t_stat = mean_val / se_val if se_val > 0 else 0
        p_value = sp_stats.t.sf(t_stat, df=n_runs - 1)  # one-sided (MI ≥ 0)

        stats.append({
            'feature': name,
            'mean': mean_val,
            'std': std_val,
            'ci_95': ci_95,
            'p_value': p_value,
        })

    return stats


def main():
    parser = argparse.ArgumentParser(description="Mutual Information Analysis")
    parser.add_argument("--model", type=str, default="rl_model", help="Path to trained model")
    parser.add_argument("--months", type=int, default=6000, help="Months per trajectory")
    parser.add_argument("--n-runs", type=int, default=30, help="Number of independent runs")
    parser.add_argument("--lags", type=int, nargs='+', default=[1, 3, 6, 12, 18, 24],
                        help="Lag values in months")
    parser.add_argument("--n-bins", type=int, default=20, help="Histogram bins for MI")
    parser.add_argument("--threshold", type=float, default=0.5, help="ENSO class threshold")
    parser.add_argument("--master-seed", type=int, default=42, help="Master random seed")
    args = parser.parse_args()

    suppress_warnings()
    np.random.seed(args.master_seed)

    output_dir = Path("plots") / args.model / "mutual_information"
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    print("=" * 70)
    print("MUTUAL INFORMATION ANALYSIS (Robust Multi-Run)")
    print("=" * 70)

    env_config = EnvConfig()
    model, env, var_names = load_environment(args.model, env_config)
    action_names = list(var_names[1:])
    n_actions = len(action_names)

    # Generate shared seeds from master RNG
    master_rng = np.random.default_rng(args.master_seed)
    shared_seeds = [int(master_rng.integers(0, 2**31)) for _ in range(args.n_runs)]

    print(f"\n  N_RUNS          = {args.n_runs}")
    print(f"  SIM_MONTHS      = {args.months} ({args.months // 12} years)")
    print(f"  LAGS            = {args.lags}")
    print(f"  THRESHOLD       = {args.threshold}")
    print(f"{'=' * 70}\n")

    # Collect per-run results
    # mi_per_run[run, lag_idx, action] — MI at each lag for each run
    mi_per_run_all_lags = np.zeros((args.n_runs, len(args.lags), n_actions))

    for run_idx, seed in enumerate(shared_seeds):
        print(f"--- Run {run_idx+1}/{args.n_runs} (seed={seed}) ---")

        lag_profiles, actions, nino34 = compute_mi_for_run(
            env, model, args.months, seed,
            args.lags, args.threshold, args.n_bins
        )
        mi_per_run_all_lags[run_idx] = lag_profiles

        # Report for primary lag
        primary_mi = lag_profiles[0]
        top_idx = np.argmax(primary_mi)
        enso_frac = np.mean(np.abs(nino34) > args.threshold) * 100
        print(f"  ENSO fraction: {enso_frac:.1f}% | "
              f"Top MI (lag={args.lags[0]}): {action_names[top_idx]} "
              f"({primary_mi[top_idx]*1000:.2f} millinats)\n")

    # Primary lag analysis
    primary_lag_idx = 0
    primary_lag = args.lags[primary_lag_idx]
    mi_primary_per_run = mi_per_run_all_lags[:, primary_lag_idx, :]  # [N_RUNS, 9]

    # Statistical analysis at primary lag
    stats = compute_statistics(mi_primary_per_run, action_names)

    print(f"\n{'='*100}")
    print(f"MUTUAL INFORMATION — Statistical Summary (lag={primary_lag}, N={args.n_runs} runs)")
    print(f"{'='*100}")
    print(f"{'Action':<12} | {'Mean MI':>12} | {'Std':>10} | {'95% CI':>28} | {'p-value':>10} | {'Sig?':>6}")
    print(f"{'-'*90}")

    for s in sorted(stats, key=lambda x: x['mean'], reverse=True):
        ci_lo = (s['mean'] - s['ci_95']) * 1000
        ci_hi = (s['mean'] + s['ci_95']) * 1000
        sig = "***" if s['p_value'] < 0.001 else "**" if s['p_value'] < 0.01 else "*" if s['p_value'] < 0.05 else "ns"
        print(f"{s['feature']:<12} | {s['mean']*1000:>10.3f} mn | {s['std']*1000:>8.3f} mn | "
              f"[{ci_lo:>+10.3f}, {ci_hi:>+10.3f}] mn | {s['p_value']:>10.4f} | {sig:>6}")

    print(f"\n(mn = millinats)")
    print(f"Significance: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
    print(f"(One-sided test: MI > 0)")

    # Mean lag profiles across runs
    mean_lag_profiles = mi_per_run_all_lags.mean(axis=0)  # [n_lags, 9]

    print(f"\n{'='*70}")
    print("LAG PROFILE (Mean MI across runs)")
    print(f"{'='*70}")
    header = f"{'Lag':>6}" + "".join(f"{name:>12}" for name in action_names)
    print(header)
    print("-" * len(header))
    for i, lag in enumerate(args.lags):
        row = f"{lag:>6}" + "".join(f"{mean_lag_profiles[i, j]*1000:>12.3f}" for j in range(n_actions))
        print(row)

    # Save results
    np.savez(
        output_dir / 'mi_results.npz',
        # v2: MI computed on RAW actions against RAW Nino3.4. v1 used step%12-scaled
        # actions and the Z-SCORED Nino3.4 thresholded in raw degC -- see
        # collect_single_trajectory. v1 numbers are not comparable to these.
        schema_version=2,
        actions='raw',
        mi_per_run_all_lags=mi_per_run_all_lags,
        mi_primary_per_run=mi_primary_per_run,
        mean_lag_profiles=mean_lag_profiles,
        lags=np.array(args.lags),
        action_names=action_names,
        var_names=var_names,
        n_runs=args.n_runs,
        months=args.months,
        seeds=shared_seeds,
        threshold=args.threshold,
    )
    print(f"\nResults saved to {output_dir / 'mi_results.npz'}")

    elapsed = time.time() - start_time
    hours, remainder = divmod(int(elapsed), 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"\n{'='*70}")
    print(f"MUTUAL INFORMATION ANALYSIS COMPLETE — Total time: {hours}h {minutes}m {seconds}s")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
