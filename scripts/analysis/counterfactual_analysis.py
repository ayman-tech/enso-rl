"""
Counterfactual Trajectory Analysis for ENSO RL Agent.

Instead of disabling actions (setting to zero), clamps each action to its
empirical mean. This removes variation while preserving the average contribution,
providing a more precise counterfactual than zero-ablation.

Uses paired seeded runs for statistical robustness.

Usage:
    uv run scripts/analysis/counterfactual_analysis.py --model rl_model
    uv run scripts/analysis/counterfactual_analysis.py --model rl-model --n-runs 30 --months 1200
    uv run scripts/analysis/counterfactual_analysis.py --model rl_model --metric mye_prob

Robust :
    uv run scripts/analysis/counterfactual_analysis.py \
        --model rl_model --metric mye_prob \
        --months 2400 --n-runs 100
"""
import sys
import time
import argparse
import multiprocessing as mp
from multiprocessing import cpu_count
import numpy as np
import matplotlib.pyplot as plt
import wandb
from datetime import datetime
from pathlib import Path
from scipy import stats as sp_stats

repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from stable_baselines3 import PPO
from config import EnvConfig, WandbConfig
from utils.data_processing import load_observational_data, prepare_xro_parameters
from utils.enso_classifier import classify_enso_event, mye_fraction_by_phase
from envs import XROMultiYearEnv
from utils import suppress_warnings
from utils.results_io import save_csv
from XRO.core import XRO

METRIC_LABELS = {
    'avg_reward': 'Average Reward',
    'mye_prob': 'MYE Probability',
}

# Map metric name to result dict keys
METRIC_KEYS = {
    'avg_reward': ('rewards_baseline', 'rewards_mean_clamp', 'rewards_zero_ablate'),
    'mye_prob': ('mye_baseline', 'mye_mean_clamp', 'mye_zero_ablate'),
}


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
    params = prepare_xro_parameters(model_xro, train_ds, var_names, bounds)
    params['threshold'] = env_config.threshold

    env = XROMultiYearEnv(
        params=params, train_ds=train_ds,
        var_names=var_names, max_steps=env_config.max_steps
    )

    model = PPO.load(str(model_path), env=env)
    return model, env, var_names


def collect_mean_actions(env, model, num_months, n_trajectories=5, seed=42):
    """
    Collect empirical mean action for each action dimension.

    Runs several trajectories and computes the per-dimension mean of the
    agent's actions.

    Args:
        env: Environment
        model: Trained PPO model
        num_months: Months per trajectory
        n_trajectories: Number of trajectories
        seed: Master seed

    Returns:
        np.ndarray: Mean action vector [9]
    """
    rng = np.random.default_rng(seed)
    all_actions = []

    for i in range(n_trajectories):
        s = int(rng.integers(0, 2**31))
        obs, _ = env.reset(seed=s)
        for step in range(num_months):
            action, _ = model.predict(obs, deterministic=True)
            all_actions.append(action.copy())
            obs, _, _, _, _ = env.step(action)

    all_actions = np.array(all_actions)
    mean_action = all_actions.mean(axis=0)
    return mean_action


def simulate_counterfactual(env, model, num_months, seed,
                            clamp_idx=None, clamp_value=None,
                            zero_idx=None):
    """
    Simulate with a counterfactual modification to one action.

    Args:
        env: Environment
        model: Trained PPO model
        num_months: Duration
        seed: Random seed
        clamp_idx: Action index to clamp to clamp_value
        clamp_value: Value to clamp the action to
        zero_idx: Action index to set to zero (classic ablation)

    Returns:
        dict with avg_reward and mye_prob
    """
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    enso_history = [obs[0]]

    for step in range(num_months):
        action, _ = model.predict(obs, deterministic=True)

        if clamp_idx is not None and clamp_value is not None:
            action[clamp_idx] = clamp_value
        if zero_idx is not None:
            action[zero_idx] = 0.0

        obs, reward, _, _, _ = env.step(action)
        total_reward += reward
        enso_history.append(obs[0])

    classified = classify_enso_event(enso_history)
    phase = mye_fraction_by_phase(classified)

    return {
        'avg_reward': total_reward / num_months,
        'mye_prob': phase['total'],
        'mye_prob_el_nino': phase['el_nino'],
        'mye_prob_la_nina': phase['la_nina'],
    }


def run_paired_analysis(env, model, var_names, num_months, n_runs,
                        mean_actions, master_seed=42):
    """
    Run paired counterfactual analysis: baseline vs mean-clamped vs zero-ablated.

    For each seed, run:
    - Baseline (full control)
    - Mean-clamped for each action (9 conditions)
    - Zero-ablated for each action (9 conditions)

    All share the same seed.

    Args:
        env: Environment
        model: Trained model
        var_names: Variable names
        num_months: Months per sim
        n_runs: Number of paired seeds
        mean_actions: Empirical mean action [9]
        master_seed: Seed generator

    Returns:
        dict with reward matrices and analysis
    """
    rng = np.random.default_rng(master_seed)
    seeds = [int(rng.integers(0, 2**31)) for _ in range(n_runs)]
    n_features = 9
    controllable_vars = var_names[1:]

    # Storage
    rewards_baseline = np.zeros(n_runs)
    rewards_mean_clamp = np.zeros((n_runs, n_features))
    rewards_zero_ablate = np.zeros((n_runs, n_features))
    mye_baseline = np.zeros(n_runs)
    mye_mean_clamp = np.zeros((n_runs, n_features))
    mye_zero_ablate = np.zeros((n_runs, n_features))

    total_sims = n_runs * (1 + 2 * n_features)
    sim_count = 0

    for run_idx, seed in enumerate(seeds):
        print(f"  Run {run_idx+1}/{n_runs} (seed={seed})")

        # Baseline
        result = simulate_counterfactual(env, model, num_months, seed)
        rewards_baseline[run_idx] = result['avg_reward']
        mye_baseline[run_idx] = result['mye_prob']
        sim_count += 1

        for feat_idx in range(n_features):
            # Mean-clamped
            result = simulate_counterfactual(
                env, model, num_months, seed,
                clamp_idx=feat_idx, clamp_value=float(mean_actions[feat_idx])
            )
            rewards_mean_clamp[run_idx, feat_idx] = result['avg_reward']
            mye_mean_clamp[run_idx, feat_idx] = result['mye_prob']
            sim_count += 1

            # Zero-ablated (classical)
            result = simulate_counterfactual(
                env, model, num_months, seed,
                zero_idx=feat_idx
            )
            rewards_zero_ablate[run_idx, feat_idx] = result['avg_reward']
            mye_zero_ablate[run_idx, feat_idx] = result['mye_prob']
            sim_count += 1

        if (run_idx + 1) % 5 == 0:
            print(f"    Progress: {sim_count}/{total_sims} simulations")

    return {
        'rewards_baseline': rewards_baseline,
        'rewards_mean_clamp': rewards_mean_clamp,
        'rewards_zero_ablate': rewards_zero_ablate,
        'mye_baseline': mye_baseline,
        'mye_mean_clamp': mye_mean_clamp,
        'mye_zero_ablate': mye_zero_ablate,
        'controllable_vars': controllable_vars,
        'seeds': seeds,
        'mean_actions': mean_actions,
    }


def compute_statistics(results, metric='mye_prob'):
    """Compute Δ statistics for both counterfactual methods."""
    baseline_key, clamp_key, zero_key = METRIC_KEYS[metric]
    n_runs = len(results[baseline_key])
    baseline = results[baseline_key]
    controllable_vars = results['controllable_vars']

    stats = []
    for feat_idx, feat_name in enumerate(controllable_vars):
        # Mean-clamped Δ
        dr_clamp = results[clamp_key][:, feat_idx] - baseline
        mean_clamp = dr_clamp.mean()
        se_clamp = dr_clamp.std(ddof=1) / np.sqrt(n_runs)
        ci_clamp = sp_stats.t.ppf(0.975, df=n_runs - 1) * se_clamp
        t_clamp = mean_clamp / se_clamp if se_clamp > 0 else 0
        p_clamp = 2 * sp_stats.t.sf(abs(t_clamp), df=n_runs - 1)

        # Zero-ablated Δ
        dr_zero = results[zero_key][:, feat_idx] - baseline
        mean_zero = dr_zero.mean()
        se_zero = dr_zero.std(ddof=1) / np.sqrt(n_runs)
        ci_zero = sp_stats.t.ppf(0.975, df=n_runs - 1) * se_zero
        t_zero = mean_zero / se_zero if se_zero > 0 else 0
        p_zero = 2 * sp_stats.t.sf(abs(t_zero), df=n_runs - 1)

        stats.append({
            'feature': feat_name,
            'dr_mean_clamp': mean_clamp,
            'ci_clamp': ci_clamp,
            'p_clamp': p_clamp,
            'dr_zero': mean_zero,
            'ci_zero': ci_zero,
            'p_zero': p_zero,
        })

    return stats


def plot_comparison(stats, n_runs, output_dir, metric='mye_prob'):
    """Plot mean-clamp vs zero-ablation comparison."""
    metric_label = METRIC_LABELS.get(metric, metric)
    features = [s['feature'] for s in stats]
    n = len(features)

    # Sort by mean-clamp Δ magnitude
    sorted_idx = np.argsort([abs(s['dr_mean_clamp']) for s in stats])[::-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8), sharey=True)

    x = np.arange(n)
    width = 0.35

    # Left: Mean-Clamped
    sorted_features = [features[i] for i in sorted_idx]
    clamp_vals = [stats[i]['dr_mean_clamp'] for i in sorted_idx]
    clamp_cis = [stats[i]['ci_clamp'] for i in sorted_idx]
    clamp_colors = ['#D32F2F' if stats[i]['p_clamp'] < 0.05 else '#999999' for i in sorted_idx]

    ax1.bar(x, clamp_vals, width=0.6, color=clamp_colors, edgecolor='black',
            yerr=clamp_cis, capsize=4, error_kw={'linewidth': 1.2})
    ax1.axhline(0, color='gray', linestyle='--')
    ax1.set_xticks(x)
    ax1.set_xticklabels(sorted_features, rotation=45, ha='right')
    ax1.set_ylabel(f'Mean Δ{metric_label} ± 95% CI (N={n_runs})', fontsize=13)
    ax1.set_title(f'Counterfactual: Clamp to Mean — {metric_label}', fontsize=15)
    ax1.grid(axis='y', alpha=0.3)

    # Significance stars
    for idx_pos, orig_idx in enumerate(sorted_idx):
        p = stats[orig_idx]['p_clamp']
        star = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        y = clamp_vals[idx_pos] + clamp_cis[idx_pos] + 0.001 if clamp_vals[idx_pos] >= 0 else clamp_vals[idx_pos] - clamp_cis[idx_pos] - 0.001
        va = 'bottom' if clamp_vals[idx_pos] >= 0 else 'top'
        ax1.text(idx_pos, y, star, ha='center', va=va, fontsize=10, fontweight='bold')

    # Right: Zero-Ablated
    zero_vals = [stats[i]['dr_zero'] for i in sorted_idx]
    zero_cis = [stats[i]['ci_zero'] for i in sorted_idx]
    zero_colors = ['#1976D2' if stats[i]['p_zero'] < 0.05 else '#999999' for i in sorted_idx]

    ax2.bar(x, zero_vals, width=0.6, color=zero_colors, edgecolor='black',
            yerr=zero_cis, capsize=4, error_kw={'linewidth': 1.2})
    ax2.axhline(0, color='gray', linestyle='--')
    ax2.set_xticks(x)
    ax2.set_xticklabels(sorted_features, rotation=45, ha='right')
    ax2.set_title(f'Classical Ablation: Set to Zero — {metric_label}', fontsize=15)
    ax2.grid(axis='y', alpha=0.3)

    for idx_pos, orig_idx in enumerate(sorted_idx):
        p = stats[orig_idx]['p_zero']
        star = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        y = zero_vals[idx_pos] + zero_cis[idx_pos] + 0.001 if zero_vals[idx_pos] >= 0 else zero_vals[idx_pos] - zero_cis[idx_pos] - 0.001
        va = 'bottom' if zero_vals[idx_pos] >= 0 else 'top'
        ax2.text(idx_pos, y, star, ha='center', va=va, fontsize=10, fontweight='bold')

    fig.suptitle(f'Counterfactual vs Classical Ablation — {metric_label}', fontsize=17, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / f'counterfactual_comparison_{metric}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {output_dir / f'counterfactual_comparison_{metric}.png'}")

    # Scatter: clamp Δ vs zero Δ
    fig2, ax3 = plt.subplots(figsize=(10, 10))
    dr_clamp_all = [s['dr_mean_clamp'] for s in stats]
    dr_zero_all = [s['dr_zero'] for s in stats]

    ax3.scatter(dr_zero_all, dr_clamp_all, s=120, c='steelblue', edgecolor='black', zorder=5)
    for i, feat in enumerate(features):
        ax3.annotate(feat, (dr_zero_all[i], dr_clamp_all[i]),
                     fontsize=10, ha='left', va='bottom', xytext=(5, 5),
                     textcoords='offset points')

    # Diagonal
    lims = [min(min(dr_zero_all), min(dr_clamp_all)) - 0.01,
            max(max(dr_zero_all), max(dr_clamp_all)) + 0.01]
    ax3.plot(lims, lims, 'r--', alpha=0.5, label='Agreement line')
    ax3.set_xlabel(f'Zero-Ablation Δ{metric_label}', fontsize=13)
    ax3.set_ylabel(f'Mean-Clamp Δ{metric_label}', fontsize=13)
    ax3.set_title(f'Method Agreement — {metric_label}', fontsize=15)
    ax3.legend()
    ax3.grid(alpha=0.3)
    ax3.set_aspect('equal')
    fig2.tight_layout()
    fig2.savefig(output_dir / f'counterfactual_scatter_{metric}.png', dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"  Saved {output_dir / f'counterfactual_scatter_{metric}.png'}")

    # Log plots to W&B
    if wandb.run is not None:
        wandb.log({
            f"counterfactual_comparison_{metric}": wandb.Image(str(output_dir / f'counterfactual_comparison_{metric}.png')),
            f"counterfactual_scatter_{metric}": wandb.Image(str(output_dir / f'counterfactual_scatter_{metric}.png')),
        })


def _phase_keys():
    return ['total', 'el_nino', 'la_nina']


def _phase_field(result, phase):
    return {'total': 'mye_prob', 'el_nino': 'mye_prob_el_nino',
            'la_nina': 'mye_prob_la_nina'}[phase]


def _worker_counterfactual(worker_args):
    """Worker: one trained seed -> mean zero-ablation ΔP(MYE) per feature per phase.

    Each worker loads its own model/env (spawn-safe); heavy imports happen here,
    not in the main process. Numerically identical to the serial path — only the
    independent per-seed work is distributed; cross-seed aggregation stays in main.
    Returns (seed, controllable_vars or None, seed_means dict or None). A missing
    model returns Nones so the parent can skip it, mirroring the serial behavior.
    """
    s, model_name, num_months, n_runs, master_seed = worker_args

    from utils import suppress_warnings
    suppress_warnings()

    env_config = EnvConfig()
    try:
        model, env, var_names = load_environment(model_name, env_config)
    except FileNotFoundError:
        return s, None, None

    seed_means = seed_mean_delta_by_phase(env, model, var_names,
                                          num_months, n_runs, master_seed)
    print(f"  [worker] seed={s} done", flush=True)
    return s, list(var_names[1:]), seed_means


def seed_mean_delta_by_phase(env, model, var_names, num_months, n_runs, master_seed):
    """One trained model: mean zero-ablation ΔP(MYE) per feature per phase.

    Zero-ablation (disable mode j's action) is the cleanest causal contrast for
    the convergence figure. Paired by seed (baseline vs ablation share the seed).
    Returns dict phase -> np.ndarray[n_features].
    """
    rng = np.random.default_rng(master_seed)
    seeds = [int(rng.integers(0, 2**31)) for _ in range(n_runs)]
    n_features = len(var_names) - 1
    phases = _phase_keys()
    deltas = {p: np.zeros((n_runs, n_features)) for p in phases}

    for ri, seed in enumerate(seeds):
        base = simulate_counterfactual(env, model, num_months, seed)
        for fi in range(n_features):
            abl = simulate_counterfactual(env, model, num_months, seed, zero_idx=fi)
            for p in phases:
                deltas[p][ri, fi] = abl[_phase_field(abl, p)] - base[_phase_field(base, p)]
    return {p: deltas[p].mean(axis=0) for p in phases}


def run_ensemble(args, output_dir):
    """Cross-seed counterfactual: per-seed mean ΔP(MYE) -> CIs across seeds, per phase.

    Seeds are independent, so the per-seed work is parallelized across processes
    (mirrors shapley_analysis). --workers=1 runs serially. The result is identical
    to the serial path: aggregation across seeds happens here, in deterministic
    seed order, regardless of worker completion order.
    """
    from scipy.stats import t as t_dist
    phases = _phase_keys()
    per_seed = {p: [] for p in phases}  # each: list over seeds of [n_features]
    controllable = None
    used = []

    n_workers = args.workers if args.workers else max(1, cpu_count() - 2)
    n_workers = min(n_workers, len(args.seeds))  # no more workers than tasks

    worker_args = [(s, f"{args.prefix}_seed{s}", args.months, args.n_runs, args.seed)
                   for s in args.seeds]

    if n_workers > 1:
        print(f"  Parallel across {len(args.seeds)} seeds with {n_workers} workers")
        with mp.Pool(processes=n_workers) as pool:
            results = pool.map(_worker_counterfactual, worker_args)
    else:
        print("  Serial (workers=1)")
        results = [_worker_counterfactual(wa) for wa in worker_args]

    # Aggregate in deterministic seed order (independent of completion order).
    by_seed = {r[0]: r for r in results}
    for s in args.seeds:
        _, controllable_s, seed_means = by_seed[s]
        if seed_means is None:
            print(f"  [skip] models/{args.prefix}_seed{s}.zip not found")
            continue
        if controllable is None:
            controllable = controllable_s
        for p in phases:
            per_seed[p].append(seed_means[p])
        used.append(s)
        print(f"  seed={s}: total ΔP top driver "
              f"{controllable[int(np.argmin(seed_means['total']))]}")

    if not used:
        raise FileNotFoundError("No ensemble models found (scripts/train_ensemble.py first).")

    n_seeds = len(used)
    save_kw = {'features': np.array(controllable), 'phases': np.array(phases),
               'seeds': np.array(used), 'n_runs': args.n_runs, 'months': args.months}
    for p in phases:
        M = np.vstack(per_seed[p])  # [n_seeds, n_features]
        mean = M.mean(axis=0)
        ci = (t_dist.ppf(0.975, df=n_seeds - 1) * M.std(axis=0, ddof=1) / np.sqrt(n_seeds)
              if n_seeds > 1 else np.zeros_like(mean))
        save_kw[f'mean_{p}'] = mean
        save_kw[f'ci_{p}'] = ci
        save_kw[f'per_seed_{p}'] = M  # [n_seeds, n_features] — for seed-stability plots
        print(f"\n  === Counterfactual ΔP(MYE) — {p} (N={n_seeds} seeds) ===")
        order = np.argsort(mean)
        for fi in order:
            print(f"    {controllable[fi]:<10} {mean[fi]:+.4f} ± {ci[fi]:.4f}")

    np.savez(output_dir / 'counterfactual_ensemble.npz', **save_kw)
    print(f"\n  Saved {output_dir / 'counterfactual_ensemble.npz'}")

    # Tidy CSVs alongside the npz (survive lost logs; easy notebook loading).
    summary_rows, per_seed_rows = [], []
    for p in phases:
        for fi, feat in enumerate(controllable):
            summary_rows.append({'phase': p, 'feature': feat,
                                 'mean_dP_MYE': float(save_kw[f'mean_{p}'][fi]),
                                 'ci95': float(save_kw[f'ci_{p}'][fi])})
            for si, s in enumerate(used):
                per_seed_rows.append({'phase': p, 'seed': int(s), 'feature': feat,
                                      'dP_MYE': float(save_kw[f'per_seed_{p}'][si, fi])})
    save_csv(output_dir / 'counterfactual_ensemble.csv', summary_rows)
    save_csv(output_dir / 'counterfactual_ensemble_per_seed.csv', per_seed_rows)


def main():
    # Use 'spawn' to avoid PyTorch fork deadlocks (matches shapley_analysis).
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(description="Counterfactual Trajectory Analysis")
    parser.add_argument("--model", type=str, default="rl_model", help="Path to trained model")
    parser.add_argument("--months", type=int, default=1200, help="Simulation months per run")
    parser.add_argument("--n-runs", type=int, default=20, help="Number of paired runs")
    parser.add_argument("--seed", type=int, default=42, help="Master seed")
    parser.add_argument("--metric", type=str, default="mye_prob",
                        choices=["avg_reward", "mye_prob"],
                        help="Metric for analysis (default: mye_prob)")
    parser.add_argument("--ensemble", action="store_true",
                        help="Aggregate zero-ablation ΔP(MYE) across trained seeds, per phase")
    parser.add_argument("--prefix", type=str, default="rl_model", help="Ensemble model prefix")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)),
                        help="Ensemble seeds")
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel workers for --ensemble across seeds "
                             "(default: cpu_count - 2; use 1 to force serial)")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable W&B logging")
    args = parser.parse_args()

    if args.ensemble:
        suppress_warnings()
        out = Path("plots") / args.prefix / "counterfactual"
        out.mkdir(parents=True, exist_ok=True)
        print("=" * 70)
        print("COUNTERFACTUAL — ENSEMBLE (cross-seed, phase-resolved)")
        print("=" * 70)
        run_ensemble(args, out)
        return

    suppress_warnings()
    np.random.seed(args.seed)

    metric_label = METRIC_LABELS[args.metric]

    output_dir = Path("plots") / args.model / "counterfactual" / args.metric
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize W&B
    if not args.no_wandb:
        wandb_config = WandbConfig()
        run_name = datetime.now().strftime(r"counter %H:%M %d-%m-%y")
        wandb.init(
            project=wandb_config.project,
            entity=wandb_config.entity,
            name=run_name,
            job_type="counterfactual-analysis",
            group="analysis",
            tags=["counterfactual", "analysis", "ablation", args.metric],
            config={
                "model": args.model,
                "months": args.months,
                "n_runs": args.n_runs,
                "master_seed": args.seed,
                "metric": args.metric,
            },
        )

    print("=" * 70)
    start_time = time.time()

    print(f"COUNTERFACTUAL TRAJECTORY ANALYSIS — {metric_label}")
    print("=" * 70)

    env_config = EnvConfig()
    model, env, var_names = load_environment(args.model, env_config)
    controllable_vars = var_names[1:]

    # Step 1: Collect empirical mean actions
    print("\nStep 1: Collecting empirical mean actions...")
    mean_actions = collect_mean_actions(env, model, num_months=args.months,
                                       n_trajectories=3, seed=args.seed)
    print("  Mean action vector:")
    for i, name in enumerate(controllable_vars):
        print(f"    {name:<10}: {mean_actions[i]:+.4f}")

    if wandb.run is not None:
        wandb.log({f"mean_action/{name}": float(mean_actions[i]) for i, name in enumerate(controllable_vars)})

    # Step 2: Run paired analysis (collects both metrics in one pass)
    print(f"\nStep 2: Running paired analysis ({args.n_runs} runs × {1 + 2*9} conditions = {args.n_runs * 19} sims)...")
    results = run_paired_analysis(
        env, model, var_names,
        num_months=args.months, n_runs=args.n_runs,
        mean_actions=mean_actions, master_seed=args.seed
    )

    # Step 3: Statistical analysis
    print(f"\nStep 3: Computing statistics for {metric_label}...")
    stats = compute_statistics(results, metric=args.metric)

    print(f"\n{'='*100}")
    print(f"COUNTERFACTUAL ANALYSIS — {metric_label} (N={args.n_runs} paired runs)")
    print(f"{'='*100}")
    print(f"{'Feature':<10} | {'Clamp Δ':>10} {'±CI':>8} {'p':>8} {'Sig':>5} | {'Zero Δ':>10} {'±CI':>8} {'p':>8} {'Sig':>5}")
    print("-" * 100)

    for s in sorted(stats, key=lambda x: x['dr_mean_clamp']):
        sig_c = "***" if s['p_clamp'] < 0.001 else "**" if s['p_clamp'] < 0.01 else "*" if s['p_clamp'] < 0.05 else "ns"
        sig_z = "***" if s['p_zero'] < 0.001 else "**" if s['p_zero'] < 0.01 else "*" if s['p_zero'] < 0.05 else "ns"
        print(f"{s['feature']:<10} | {s['dr_mean_clamp']:>+10.4f} {s['ci_clamp']:>8.4f} {s['p_clamp']:>8.4f} {sig_c:>5} | {s['dr_zero']:>+10.4f} {s['ci_zero']:>8.4f} {s['p_zero']:>8.4f} {sig_z:>5}")

    # Log summary table to W&B
    if wandb.run is not None:
        summary_table = wandb.Table(
            columns=["Feature", "Clamp Δ", "Clamp CI", "Clamp p-value", "Clamp Sig",
                     "Zero Δ", "Zero CI", "Zero p-value", "Zero Sig"],
            data=[
                [
                    s['feature'],
                    s['dr_mean_clamp'], s['ci_clamp'], s['p_clamp'],
                    "***" if s['p_clamp'] < 0.001 else "**" if s['p_clamp'] < 0.01 else "*" if s['p_clamp'] < 0.05 else "ns",
                    s['dr_zero'], s['ci_zero'], s['p_zero'],
                    "***" if s['p_zero'] < 0.001 else "**" if s['p_zero'] < 0.01 else "*" if s['p_zero'] < 0.05 else "ns",
                ]
                for s in sorted(stats, key=lambda x: abs(x['dr_mean_clamp']), reverse=True)
            ]
        )
        wandb.log({f"counterfactual_summary_{args.metric}": summary_table})

    # Agreement check
    from scipy.stats import spearmanr
    clamp_ranks = [s['dr_mean_clamp'] for s in stats]
    zero_ranks = [s['dr_zero'] for s in stats]
    rho, pval = spearmanr(clamp_ranks, zero_ranks)
    print(f"\nRank correlation (Spearman) between methods: ρ = {rho:.3f}, p = {pval:.4f}")

    if wandb.run is not None:
        wandb.log({f"spearman_rho_{args.metric}": rho, f"spearman_p_{args.metric}": pval})

    # Save
    baseline_key, clamp_key, zero_key = METRIC_KEYS[args.metric]
    np.savez(
        output_dir / 'counterfactual_results.npz',
        baseline=results[baseline_key],
        mean_clamp=results[clamp_key],
        zero_ablate=results[zero_key],
        mean_actions=mean_actions,
        controllable_vars=controllable_vars,
        metric=args.metric,
    )
    print(f"\nResults saved to {output_dir / 'counterfactual_results.npz'}")

    # Plot
    print("\nGenerating plots...")
    plot_comparison(stats, args.n_runs, output_dir, metric=args.metric)

    if wandb.run is not None:
        wandb.finish()

    elapsed = time.time() - start_time
    hours, remainder = divmod(int(elapsed), 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"\n{'='*70}")
    print(f"COUNTERFACTUAL ANALYSIS COMPLETE — Total time: {hours}h {minutes}m {seconds}s")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
