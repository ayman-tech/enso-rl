"""
Shapley Value Analysis for ENSO RL Agent Actions (ensemble mode).

Computes permutation-sampling Shapley values aggregated across independently
trained seeds; saves results for notebook plotting.

Usage:
    uv run scripts/analysis/shapley_analysis.py --model ensemble \
        --seeds 0 1 2 3 4 5 6 7 8 9 --n-runs 30 --months 1200
"""
import sys
import os
import time
import argparse
import numpy as np
import wandb
from datetime import datetime
from pathlib import Path
from scipy import stats as sp_stats
import multiprocessing as mp
from multiprocessing import cpu_count

# Add repo root to path (must precede any repo-local imports)
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from utils import suppress_warnings
from utils.results_io import save_csv

# Fixed variable names — avoids loading PyTorch in main process before fork
ACTION_NAMES = ['WWV', 'NPMM', 'SPMM', 'IOB', 'IOD', 'SIOD', 'TNA', 'ATL3', 'SASD']

METRIC_LABELS = {
    'mye_prob': 'MYE Probability',
    'enso_months': 'Total ENSO Months',
    'avg_reward': 'Average Reward',
}


def compute_metric_from_trajectory(enso_history, threshold, metric):
    """Compute a scalar metric from an ENSO trajectory.

    Args:
        enso_history: List/array of Nino3.4 values
        threshold: ENSO event threshold
        metric: One of 'mye_prob', 'enso_months'

    Returns:
        float: Metric value
    """
    from utils.enso_classifier import classify_enso_event, mye_fraction_by_phase
    enso = np.array(enso_history)

    if metric in ('mye_prob', 'mye_prob_el_nino', 'mye_prob_la_nina'):
        classified = classify_enso_event(enso, threshold=threshold)
        phase = mye_fraction_by_phase(classified)
        return {'mye_prob': phase['total'], 'mye_prob_el_nino': phase['el_nino'],
                'mye_prob_la_nina': phase['la_nina']}[metric]

    if metric == '_unused_legacy_mye':
        classified = classify_enso_event(enso, threshold=threshold)
        mye_months = np.sum(
            (classified == 'Multi-year El Nino') | (classified == 'Multi-year La Nina')
        )
        return mye_months / len(classified)

    elif metric == 'enso_months':
        return float(np.sum(np.abs(enso) >= threshold))

    raise ValueError(f"Unknown metric: {metric}")


def load_environment(model_path: str):
    """Load trained model and create environment."""
    from stable_baselines3 import PPO
    from config import EnvConfig
    from utils.data_processing import load_observational_data, prepare_xro_parameters
    from envs import XROMultiYearEnv
    from XRO.core import XRO

    env_config = EnvConfig()
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
    params = prepare_xro_parameters(model_xro, train_ds, var_names, bounds)
    params['threshold'] = env_config.threshold

    env = XROMultiYearEnv(
        params=params, train_ds=train_ds,
        var_names=var_names, max_steps=env_config.max_steps
    )

    model = PPO.load(str(model_path), env=env)
    return model, env


def simulate_with_coalition(env, model, coalition_mask, num_months, seed, metric):
    """
    Simulate with only a subset of actions active.
    Actions outside the coalition are clamped to zero.

    Args:
        env: Environment
        model: Trained PPO model
        coalition_mask: Boolean array [9]. True = action active.
        num_months: Simulation duration
        seed: Random seed for env.reset()
        metric: Value function - 'mye_prob', 'enso_months', or 'avg_reward'

    Returns:
        float: Metric value from the simulation
    """
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    enso_history = [float(obs[0])]

    for step in range(num_months):
        action, _ = model.predict(obs, deterministic=True)
        action = action * coalition_mask.astype(np.float32)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        enso_history.append(float(obs[0]))

    if metric == 'avg_reward':
        return total_reward / num_months

    return compute_metric_from_trajectory(enso_history, env.threshold, metric)


def compute_shapley_for_seed(env, model, num_months, n_permutations, seed, metric='mye_prob'):
    """
    Compute Shapley values for a single seed via permutation sampling.

    Args:
        env: Environment
        model: Trained model
        num_months: Months per simulation
        n_permutations: Number of random permutations
        seed: Seed for env.reset() (shared across all coalition evaluations)
        metric: Value function metric

    Returns:
        np.ndarray: Shapley values for this seed [9]
        np.ndarray: Per-permutation marginals [n_permutations, 9]
    """
    n_actions = 9
    all_marginals = np.zeros((n_permutations, n_actions))

    for perm_idx in range(n_permutations):
        perm = np.random.permutation(n_actions)

        prev_value = simulate_with_coalition(
            env, model,
            coalition_mask=np.zeros(n_actions, dtype=bool),
            num_months=num_months, seed=seed, metric=metric
        )

        for pos in range(n_actions):
            feature_idx = perm[pos]
            coalition = np.zeros(n_actions, dtype=bool)
            coalition[perm[:pos + 1]] = True

            current_value = simulate_with_coalition(
                env, model, coalition_mask=coalition,
                num_months=num_months, seed=seed, metric=metric
            )
            all_marginals[perm_idx, feature_idx] = current_value - prev_value
            prev_value = current_value

    shapley_values = all_marginals.mean(axis=0)
    return shapley_values, all_marginals


def _worker_shapley(args):
    """
    Worker function for multiprocessing.
    Each worker loads its own model/env to avoid shared state issues.
    All heavy imports (PyTorch, SB3) happen here, not in the main process.
    """
    run_idx, seed, model_path, num_months, n_permutations, metric, perm_seed = args

    from utils import suppress_warnings
    suppress_warnings()
    np.random.seed(perm_seed)

    model, env = load_environment(model_path)

    sv, marginals = compute_shapley_for_seed(
        env, model, num_months, n_permutations, seed, metric=metric
    )

    top_idx = np.argmax(np.abs(sv))
    print(f"  [Worker] Run {run_idx+1} (seed={seed}) done — top: {ACTION_NAMES[top_idx]} (SV={sv[top_idx]:+.6f})", flush=True)

    return run_idx, sv, marginals


def compute_statistics(shapley_per_run, action_names):
    """
    Compute paired-run statistics on Shapley values.

    Args:
        shapley_per_run: [N_RUNS, 9] Shapley values per run
        action_names: list of action names

    Returns:
        list of dicts with per-feature statistics
    """
    n_runs = shapley_per_run.shape[0]
    stats = []

    for i, name in enumerate(action_names):
        values = shapley_per_run[:, i]
        mean_sv = values.mean()
        std_sv = values.std(ddof=1)
        se_sv = std_sv / np.sqrt(n_runs)
        ci_95 = sp_stats.t.ppf(0.975, df=n_runs - 1) * se_sv

        # t-test: is Shapley value significantly different from 0?
        t_stat = mean_sv / se_sv if se_sv > 0 else 0
        p_value = 2 * sp_stats.t.sf(abs(t_stat), df=n_runs - 1)

        stats.append({
            'feature': name,
            'mean': mean_sv,
            'std': std_sv,
            'ci_95': ci_95,
            'p_value': p_value,
        })

    return stats


PHASE_METRICS = {'total': 'mye_prob', 'el_nino': 'mye_prob_el_nino',
                 'la_nina': 'mye_prob_la_nina'}


def _worker_shapley_ensemble(worker_args):
    """Worker: one trained seed -> mean Shapley per feature, for every phase.

    Each worker loads its own model/env (spawn-safe) and seeds np.random
    deterministically from the trained seed, so permutation sampling is
    reproducible and INDEPENDENT of execution order. That determinism is what
    makes the ensemble parallelizable: the previous version threaded a single
    global RNG through nested loops (order-dependent), so it could not be split
    across processes. The Shapley methodology, env seeds, permutation count, and
    per-phase handling are all unchanged; only the permutation stream is now
    seeded per trained seed instead of one continuous global stream. Returns
    (trained_seed, {phase: mean_sv}) or (seed, None) if the model is missing.
    """
    s, model_name, months, n_runs, n_permutations, master_seed, env_seeds = worker_args

    from utils import suppress_warnings
    suppress_warnings()
    # Deterministic, order-independent permutation stream for this trained seed.
    np.random.seed(master_seed + 1 + s)

    try:
        model, env = load_environment(model_name)
    except FileNotFoundError:
        return s, None

    phases = list(PHASE_METRICS.keys())
    feat = ACTION_NAMES
    out = {}
    for p in phases:
        sv_runs = np.zeros((n_runs, len(feat)))
        for ri, es in enumerate(env_seeds):
            sv, _ = compute_shapley_for_seed(env, model, months,
                                             n_permutations, es,
                                             metric=PHASE_METRICS[p])
            sv_runs[ri] = sv
        out[p] = sv_runs.mean(axis=0)
    print(f"  [worker] seed={s} done", flush=True)
    return s, out


def run_ensemble(args, output_dir):
    """Cross-seed, phase-resolved Shapley: per-seed mean SV -> CIs across seeds.

    Trained seeds are independent, so per-seed work is parallelized across
    processes (--workers; default cpu_count-2, 1 = serial). Permutation sampling
    is seeded deterministically per trained seed inside each worker, so the
    serial and parallel paths produce identical results; aggregation across
    seeds happens here in deterministic seed order.
    Saves shapley_ensemble.npz for the convergence figure (Part D).
    """
    from scipy.stats import t as t_dist
    phases = list(PHASE_METRICS.keys())
    per_seed = {p: [] for p in phases}
    used = []
    feat = ACTION_NAMES

    env_rng = np.random.default_rng(args.master_seed)
    env_seeds = [int(env_rng.integers(0, 2**31)) for _ in range(args.n_runs)]

    n_workers = args.workers if args.workers else max(1, cpu_count() - 2)
    n_workers = min(n_workers, len(args.seeds))  # no more workers than tasks

    worker_args = [(s, f"{args.model}_seed{s}", args.months, args.n_runs,
                    args.n_permutations, args.master_seed, env_seeds)
                   for s in args.seeds]

    if n_workers > 1:
        print(f"  Parallel across {len(args.seeds)} seeds with {n_workers} workers", flush=True)
        with mp.Pool(processes=n_workers) as pool:
            results = pool.map(_worker_shapley_ensemble, worker_args)
    else:
        print("  Serial (workers=1)", flush=True)
        results = [_worker_shapley_ensemble(wa) for wa in worker_args]

    # Aggregate in deterministic seed order (independent of completion order).
    by_seed = {r[0]: r for r in results}
    for s in args.seeds:
        _, out = by_seed[s]
        if out is None:
            print(f"  [skip] models/{args.model}_seed{s}.zip not found", flush=True)
            continue
        for p in phases:
            per_seed[p].append(out[p])
        used.append(s)
        print(f"  seed={s} done", flush=True)

    if not used:
        raise FileNotFoundError("No ensemble models found (scripts/train_ensemble.py first).")

    n_seeds = len(used)
    save_kw = {'features': np.array(feat), 'phases': np.array(phases),
               'seeds': np.array(used), 'n_runs': args.n_runs,
               'months': args.months, 'n_permutations': args.n_permutations}
    from scipy.stats import ttest_1samp
    for p in phases:
        M = np.vstack(per_seed[p])
        mean = M.mean(axis=0)
        ci = (t_dist.ppf(0.975, df=n_seeds - 1) * M.std(axis=0, ddof=1) / np.sqrt(n_seeds)
              if n_seeds > 1 else np.zeros_like(mean))
        pvals = (np.array([ttest_1samp(M[:, fi], 0).pvalue for fi in range(M.shape[1])])
                 if n_seeds > 1 else np.ones(M.shape[1]))
        save_kw[f'mean_{p}'] = mean
        save_kw[f'ci_{p}'] = ci
        save_kw[f'p_{p}'] = pvals
        save_kw[f'per_seed_{p}'] = M  # [n_seeds, n_features] — for seed-stability plots
        print(f"\n  === Shapley ΔP(MYE) — {p} (N={n_seeds} seeds) ===")
        for fi in np.argsort(mean)[::-1]:
            sig = "***" if pvals[fi] < 0.001 else "**" if pvals[fi] < 0.01 else "*" if pvals[fi] < 0.05 else "ns"
            print(f"    {feat[fi]:<10} {mean[fi]:+.5f} ± {ci[fi]:.5f}  {sig}")

    np.savez(output_dir / 'shapley_ensemble.npz', **save_kw)
    print(f"\n  Saved {output_dir / 'shapley_ensemble.npz'}")

    # Tidy CSVs alongside the npz (survive lost logs; easy notebook loading).
    summary_rows, per_seed_rows = [], []
    for p in phases:
        for fi, feature in enumerate(feat):
            summary_rows.append({'phase': p, 'feature': feature,
                                 'mean_shapley': float(save_kw[f'mean_{p}'][fi]),
                                 'ci95': float(save_kw[f'ci_{p}'][fi]),
                                 'p': float(save_kw[f'p_{p}'][fi])})
            for si, s in enumerate(used):
                per_seed_rows.append({'phase': p, 'seed': int(s), 'feature': feature,
                                      'shapley': float(save_kw[f'per_seed_{p}'][si, fi])})
    save_csv(output_dir / 'shapley_ensemble.csv', summary_rows)
    save_csv(output_dir / 'shapley_ensemble_per_seed.csv', per_seed_rows)


def main():
    # Use 'spawn' to avoid PyTorch fork deadlocks on Linux
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(description="Shapley Value Analysis (ensemble)")
    parser.add_argument("--model", type=str, default="ensemble", help="Ensemble model prefix")
    parser.add_argument("--months", type=int, default=600, help="Simulation months per evaluation")
    parser.add_argument("--n-runs", type=int, default=30, help="Independent paired runs per seed")
    parser.add_argument("--n-permutations", type=int, default=20, help="Permutations per run")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)),
                        help="Ensemble seeds")
    parser.add_argument("--master-seed", type=int, default=42, help="Master random seed")
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel workers (default: cpu_count - 2; use 1 for serial)")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable W&B logging")
    args = parser.parse_args()

    suppress_warnings()
    np.random.seed(args.master_seed)
    out = Path("plots") / args.model / "shapley"
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 70, flush=True)
    print("SHAPLEY — ENSEMBLE (cross-seed, phase-resolved)", flush=True)
    print("=" * 70, flush=True)
    start_time = time.time()

    run_ensemble(args, out)

    elapsed = time.time() - start_time
    hours, remainder = divmod(int(elapsed), 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\n{'='*70}")
    print(f"SHAPLEY ANALYSIS COMPLETE — Total time: {hours}h {minutes}m {seconds}s")
    print(f"{'='*70}")

    elapsed = time.time() - start_time
    hours, remainder = divmod(int(elapsed), 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"\n{'=' * 70}")
    print(f"SHAPLEY ANALYSIS COMPLETE — Total time: {hours}h {minutes}m {seconds}s")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
