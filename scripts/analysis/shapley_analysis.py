"""
Shapley Value Analysis for ENSO RL Agent Actions.

Computes Shapley values for each action dimension using sampling-based
approximation (permutation sampling). Uses N independent paired-seed runs
for statistical robustness with t-tests, 95% CIs, and significance testing.
Uses multiprocessing to parallelize across independent seed runs.

Usage:
    uv run scripts/shapley_analysis.py --model rl_model
    uv run scripts/shapley_analysis.py --model rl_model --months 1200 --metric mye_prob --n-runs 30 --n-permutations 20
    uv run scripts/shapley_analysis.py --model rl_model --no-wandb
Params meaning :
    n-runs: No of independent seeded simulation runs for statistical robustness (each produces one set of Shapley values; used for t-tests and confidence intervals).
    n-permutations: No of random action orderings sampled/run to approximate the Shapley values (more = better approximation of the combinatorial sum).
"""
import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import wandb
from datetime import datetime
from pathlib import Path
from scipy import stats as sp_stats
from multiprocessing import Pool, cpu_count

# Add repo root to path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from stable_baselines3 import PPO
from config import EnvConfig, WandbConfig
from utils.data_processing import load_observational_data, prepare_xro_parameters
from utils.enso_classifier import classify_enso_event
from envs import XROMultiYearEnv
from utils import suppress_warnings
from XRO.core import XRO

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
    enso = np.array(enso_history)

    if metric == 'mye_prob':
        classified = classify_enso_event(enso, threshold=threshold)
        mye_months = np.sum(
            (classified == 'Multi-year El Nino') | (classified == 'Multi-year La Nina')
        )
        return mye_months / len(classified)

    elif metric == 'enso_months':
        return float(np.sum(np.abs(enso) >= threshold))

    raise ValueError(f"Unknown metric: {metric}")


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


def compute_shapley_for_seed(env, model, num_months, n_permutations, seed, metric='avg_reward'):
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
    """
    run_idx, seed, model_path, num_months, n_permutations, metric, perm_seed = args

    suppress_warnings()
    np.random.seed(perm_seed)

    env_config = EnvConfig()
    model, env, var_names = load_environment(model_path, env_config)

    sv, marginals = compute_shapley_for_seed(
        env, model, num_months, n_permutations, seed, metric=metric
    )

    action_names = list(var_names[1:])
    top_idx = np.argmax(np.abs(sv))
    print(f"  [Worker] Run {run_idx+1} (seed={seed}) done — top: {action_names[top_idx]} (SV={sv[top_idx]:+.6f})")

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


def plot_shapley_values(stats, shapley_per_run, action_names, n_runs, output_dir, metric='mye_prob'):
    """Generate Shapley value plots with significance annotations."""
    metric_label = METRIC_LABELS.get(metric, metric)

    # Sort by absolute mean Shapley value
    sorted_idx = np.argsort([abs(s['mean']) for s in stats])[::-1]

    sorted_names = [stats[i]['feature'] for i in sorted_idx]
    sorted_means = [stats[i]['mean'] for i in sorted_idx]
    sorted_cis = [stats[i]['ci_95'] for i in sorted_idx]
    sorted_pvals = [stats[i]['p_value'] for i in sorted_idx]

    # Color by significance & direction
    colors = []
    edge_colors = []
    for s_idx in sorted_idx:
        s = stats[s_idx]
        if s['p_value'] >= 0.05:
            colors.append('#999999')
            edge_colors.append('#CCCCCC')
        elif s['mean'] > 0:
            colors.append('#D32F2F')
            edge_colors.append('black')
        else:
            colors.append('#1976D2')
            edge_colors.append('black')

    # --- Bar Chart with CIs + significance stars ---
    fig, ax = plt.subplots(figsize=(14, 7))
    bars = ax.bar(sorted_names, sorted_means, color=colors, edgecolor=edge_colors,
                  linewidth=1.5, yerr=sorted_cis, capsize=6,
                  error_kw={'linewidth': 1.5, 'capthick': 1.5})

    for i, (mean_val, ci_val, p_val) in enumerate(zip(sorted_means, sorted_cis, sorted_pvals)):
        star = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
        y_pos = mean_val + ci_val + 0.001 if mean_val >= 0 else mean_val - ci_val - 0.001
        va = 'bottom' if mean_val >= 0 else 'top'
        ax.text(i, y_pos, star, ha='center', va=va, fontsize=11, fontweight='bold')

    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Action Variable', fontsize=13)
    ax.set_ylabel(f'Mean Shapley Value ± 95% CI (N={n_runs})', fontsize=13)
    ax.set_title(f'Shapley Analysis: Action Importance — {metric_label}', fontsize=15)
    ax.set_xticklabels(sorted_names, rotation=45, ha='right', fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#D32F2F', edgecolor='black', label='Significant positive (p<0.05)'),
        Patch(facecolor='#1976D2', edgecolor='black', label='Significant negative (p<0.05)'),
        Patch(facecolor='#999999', edgecolor='#CCCCCC', label='Not significant (p≥0.05)'),
    ]
    ax.legend(handles=legend_elements, fontsize=11)
    fig.tight_layout()
    fig.savefig(output_dir / 'shapley_values_bar.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {output_dir / 'shapley_values_bar.png'}")

    # --- Boxplot of per-run Shapley values ---
    fig2, ax2 = plt.subplots(figsize=(14, 7))
    bp = ax2.boxplot(
        [shapley_per_run[:, i] for i in sorted_idx],
        labels=sorted_names, patch_artist=True, vert=True
    )
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax2.set_xlabel('Action Variable', fontsize=13)
    ax2.set_ylabel(f'Shapley Value (N={n_runs} independent runs)', fontsize=13)
    ax2.set_title(f'Distribution of Shapley Values — {metric_label}', fontsize=15)
    ax2.set_xticklabels(sorted_names, rotation=45, ha='right', fontsize=11)
    ax2.grid(axis='y', alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(output_dir / 'shapley_values_dist.png', dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"  Saved {output_dir / 'shapley_values_dist.png'}")

    # Log plots to W&B
    if wandb.run is not None:
        wandb.log({
            "shapley_bar_chart": wandb.Image(str(output_dir / 'shapley_values_bar.png')),
            "shapley_distribution": wandb.Image(str(output_dir / 'shapley_values_dist.png')),
        })


def main():
    parser = argparse.ArgumentParser(description="Shapley Value Analysis for ENSO RL Agent")
    parser.add_argument("--model", type=str, default="rl_model", help="Path to trained model")
    parser.add_argument("--months", type=int, default=600, help="Simulation months per evaluation")
    parser.add_argument("--n-runs", type=int, default=30, help="Number of independent seeds (paired trials)")
    parser.add_argument("--n-permutations", type=int, default=20, help="Permutations per run")
    parser.add_argument("--metric", type=str, default="avg_reward",
                        choices=["mye_prob", "enso_months", "avg_reward"],
                        help="Value function for Shapley analysis (default: avg_reward)")
    parser.add_argument("--master-seed", type=int, default=42, help="Master random seed")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: cpu_count - 1)")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable W&B logging (useful for HPC without internet)")
    args = parser.parse_args()

    suppress_warnings()
    np.random.seed(args.master_seed)

    output_dir = Path("plots/shapley") / args.metric
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize W&B
    n_workers = args.workers if args.workers else max(1, cpu_count() - 2)

    if not args.no_wandb:
        wandb_config = WandbConfig()
        run_name = datetime.now().strftime(r"shapely %H:%M %d-%m-%y")
        wandb.init(
            project=wandb_config.project,
            entity=wandb_config.entity,
            name=run_name,
            job_type="shapley-analysis",
            group="analysis",
            tags=["shapley", "analysis", args.metric],
            config={
                "model": args.model,
                "months": args.months,
                "n_runs": args.n_runs,
                "n_permutations": args.n_permutations,
                "metric": args.metric,
                "master_seed": args.master_seed,
                "n_workers": n_workers,
            },
        )

    print("=" * 70)
    print("SHAPLEY VALUE ANALYSIS FOR ENSO RL AGENT")
    print("=" * 70)

    env_config = EnvConfig()
    _, _, var_names = load_environment(args.model, env_config)
    action_names = list(var_names[1:])

    # Generate shared seeds from master RNG
    master_rng = np.random.default_rng(args.master_seed)
    shared_seeds = [int(master_rng.integers(0, 2**31)) for _ in range(args.n_runs)]

    # Generate independent permutation seeds for each worker (reproducibility)
    perm_rng = np.random.default_rng(args.master_seed + 1)
    perm_seeds = [int(perm_rng.integers(0, 2**31)) for _ in range(args.n_runs)]

    n_actions = 9
    total_sims = args.n_runs * args.n_permutations * (n_actions + 1)
    print(f"\n  N_RUNS          = {args.n_runs}")
    print(f"  N_PERMUTATIONS  = {args.n_permutations}")
    print(f"  METRIC          = {args.metric} ({METRIC_LABELS[args.metric]})")
    print(f"  SIM_MONTHS      = {args.months} ({args.months // 12} years)")
    print(f"  Total sims      = {total_sims}")
    print(f"  Workers         = {n_workers}")
    print(f"  W&B             = {'enabled' if not args.no_wandb else 'disabled'}")
    print(f"{'=' * 70}\n")

    # Build worker args
    worker_args = [
        (run_idx, seed, args.model, args.months, args.n_permutations, args.metric, perm_seeds[run_idx])
        for run_idx, seed in enumerate(shared_seeds)
    ]

    # Run in parallel
    shapley_per_run = np.zeros((args.n_runs, n_actions))

    with Pool(processes=n_workers) as pool:
        results = pool.map(_worker_shapley, worker_args)

    for run_idx, sv, marginals in results:
        shapley_per_run[run_idx] = sv

        # Log per-run Shapley values to W&B
        if wandb.run is not None:
            run_log = {f"shapley/{name}": sv[i] for i, name in enumerate(action_names)}
            run_log["run_idx"] = run_idx
            wandb.log(run_log)

    # Statistical analysis
    stats = compute_statistics(shapley_per_run, action_names)

    print(f"\n{'='*100}")
    print(f"PAIRED SHAPLEY ANALYSIS — Statistical Summary (N={args.n_runs} runs)")
    print(f"{'='*100}")
    print(f"{'Feature':<10} | {'Mean SV':>10} | {'Std':>8} | {'95% CI':>20} | {'p-value':>10} | {'Sig?':>6}")
    print(f"{'-'*80}")

    for s in sorted(stats, key=lambda x: abs(x['mean']), reverse=True):
        ci_lo = s['mean'] - s['ci_95']
        ci_hi = s['mean'] + s['ci_95']
        sig = "***" if s['p_value'] < 0.001 else "**" if s['p_value'] < 0.01 else "*" if s['p_value'] < 0.05 else "ns"
        print(f"{s['feature']:<10} | {s['mean']:>+10.6f} | {s['std']:>8.6f} | [{ci_lo:>+9.6f}, {ci_hi:>+9.6f}] | {s['p_value']:>10.4f} | {sig:>6}")

    print(f"\nSum of Shapley values (mean): {shapley_per_run.mean(axis=0).sum():.6f}")
    print("(Should ≈ V(all actions) - V(no actions))")
    print(f"\nSignificance: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")

    # Log summary statistics to W&B
    if wandb.run is not None:
        summary_table = wandb.Table(
            columns=["Feature", "Mean SV", "Std", "CI Low", "CI High", "p-value", "Significance"],
            data=[
                [
                    s['feature'], s['mean'], s['std'],
                    s['mean'] - s['ci_95'], s['mean'] + s['ci_95'],
                    s['p_value'],
                    "***" if s['p_value'] < 0.001 else "**" if s['p_value'] < 0.01 else "*" if s['p_value'] < 0.05 else "ns"
                ]
                for s in sorted(stats, key=lambda x: abs(x['mean']), reverse=True)
            ]
        )
        wandb.log({"shapley_summary": summary_table})

    # Save
    np.savez(
        output_dir / 'shapley_results.npz',
        shapley_per_run=shapley_per_run,
        action_names=action_names,
        n_runs=args.n_runs,
        n_permutations=args.n_permutations,
        months=args.months,
        metric=args.metric,
        seeds=shared_seeds,
    )
    print(f"\nResults saved to {output_dir / 'shapley_results.npz'}")

    # Plot
    print("\nGenerating plots...")
    plot_shapley_values(stats, shapley_per_run, action_names, args.n_runs, output_dir, metric=args.metric)

    if wandb.run is not None:
        wandb.finish()

    print(f"\n{'=' * 70}")
    print("SHAPLEY ANALYSIS COMPLETE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
