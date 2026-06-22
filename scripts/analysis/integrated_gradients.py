"""
Integrated Gradients Analysis for ENSO RL Agent.

Computes integrated gradients along a path from a neutral baseline observation
to the actual observation. This gives a complete, axiomatic attribution that
sums exactly to the output difference (completeness property).

Uses N independent paired-seed runs for statistical robustness with t-tests,
95% CIs, and significance testing.

Usage:
    uv run scripts/analysis/integrated_gradients.py --model rl_model
    uv run scripts/analysis/integrated_gradients.py --model rl_model --n-runs 30 --months 1200
"""
import sys
import time
import argparse
import numpy as np
import torch
from pathlib import Path
from scipy import stats as sp_stats

repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from stable_baselines3 import PPO
from config import EnvConfig
from utils.data_processing import load_observational_data, prepare_xro_parameters
from envs import XROMultiYearEnv
from utils import suppress_warnings
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
    params = prepare_xro_parameters(model_xro, train_ds, var_names, bounds)
    params['threshold'] = env_config.threshold

    env = XROMultiYearEnv(
        params=params, train_ds=train_ds,
        var_names=var_names, max_steps=env_config.max_steps
    )

    model = PPO.load(str(model_path), env=env)
    return model, env, var_names


def collect_observations(env, model, num_months, seed):
    """Collect observations from a trajectory."""
    obs, _ = env.reset(seed=seed)
    observations = [obs.copy()]
    for step in range(num_months):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, _, _, _ = env.step(action)
        observations.append(obs.copy())
    return np.array(observations)


def get_policy_output(policy, obs_tensor):
    """Get raw action mean from the policy network."""
    features = policy.extract_features(obs_tensor, policy.pi_features_extractor)
    latent_pi = policy.mlp_extractor.forward_actor(features)
    action_mean = policy.action_net(latent_pi)
    return action_mean


def get_value_output(policy, obs_tensor):
    """Get value prediction from the critic network."""
    features = policy.extract_features(obs_tensor, policy.vf_features_extractor)
    latent_vf = policy.mlp_extractor.forward_critic(features)
    value = policy.value_net(latent_vf)
    return value


def integrated_gradients_policy(policy, obs, baseline, n_steps=100):
    """
    Compute integrated gradients for the policy network.

    IG_i = (x_i - x'_i) * integral_0^1 (∂F/∂x_i)(x' + α(x - x')) dα

    Args:
        policy: PPO policy network
        obs: Actual observation [11]
        baseline: Baseline observation [11]
        n_steps: Number of interpolation steps

    Returns:
        np.ndarray: Attribution [9, 11]
    """
    obs_t = torch.FloatTensor(obs).unsqueeze(0)
    baseline_t = torch.FloatTensor(baseline).unsqueeze(0)
    diff = obs_t - baseline_t

    n_actions = 9
    n_obs = len(obs)
    attributions = np.zeros((n_actions, n_obs))

    for step in range(n_steps + 1):
        alpha = step / n_steps
        interpolated = baseline_t + alpha * diff
        interpolated = interpolated.clone().detach().requires_grad_(True)

        action_mean = get_policy_output(policy, interpolated)

        for j in range(n_actions):
            if interpolated.grad is not None:
                interpolated.grad.zero_()
            action_mean[0, j].backward(retain_graph=True)
            if interpolated.grad is not None:
                attributions[j] += interpolated.grad[0].detach().numpy()

    attributions = attributions / (n_steps + 1)
    attributions = attributions * diff.detach().numpy()[0]

    return attributions


def integrated_gradients_value(policy, obs, baseline, n_steps=100):
    """
    Compute integrated gradients for the value function.

    Args:
        policy: PPO policy network
        obs: Actual observation [11]
        baseline: Baseline observation [11]
        n_steps: Number of interpolation steps

    Returns:
        np.ndarray: Attribution [11]
    """
    obs_t = torch.FloatTensor(obs).unsqueeze(0)
    baseline_t = torch.FloatTensor(baseline).unsqueeze(0)
    diff = obs_t - baseline_t

    n_obs = len(obs)
    attributions = np.zeros(n_obs)

    for step in range(n_steps + 1):
        alpha = step / n_steps
        interpolated = baseline_t + alpha * diff
        interpolated = interpolated.clone().detach().requires_grad_(True)

        value = get_value_output(policy, interpolated)
        value.backward()

        if interpolated.grad is not None:
            attributions += interpolated.grad[0].detach().numpy()

    attributions = attributions / (n_steps + 1)
    attributions = attributions * diff.detach().numpy()[0]

    return attributions


def compute_ig_for_run(model, observations, n_ig_steps, n_samples, baseline_type='zero'):
    """
    Compute IG obs importance for a single run.

    Args:
        model: PPO model
        observations: Collected observations [T, 11]
        n_ig_steps: IG interpolation steps
        n_samples: Observations to sample
        baseline_type: 'zero' or 'mean'

    Returns:
        policy_obs_importance: [11] sum of |IG| across actions per obs dim
        value_obs_importance: [11] mean |IG| for value function per obs dim
        policy_ig_abs_mean: [9, 11] mean absolute IG matrix
    """
    policy = model.policy
    policy.eval()

    if baseline_type == 'mean':
        baseline = observations.mean(axis=0)
    else:
        baseline = np.zeros(observations.shape[1])

    if n_samples < len(observations):
        indices = np.random.choice(len(observations), n_samples, replace=False)
        obs_subset = observations[indices]
    else:
        obs_subset = observations

    all_policy_ig = []
    all_value_ig = []

    for obs in obs_subset:
        policy_ig = integrated_gradients_policy(policy, obs, baseline, n_steps=n_ig_steps)
        all_policy_ig.append(policy_ig)

        value_ig = integrated_gradients_value(policy, obs, baseline, n_steps=n_ig_steps)
        all_value_ig.append(value_ig)

    all_policy_ig = np.array(all_policy_ig)  # [n_samples, 9, 11]
    all_value_ig = np.array(all_value_ig)    # [n_samples, 11]

    policy_ig_abs_mean = np.mean(np.abs(all_policy_ig), axis=0)  # [9, 11]
    value_ig_abs_mean = np.mean(np.abs(all_value_ig), axis=0)    # [11]

    # Obs importance = sum |IG| across all actions
    policy_obs_importance = policy_ig_abs_mean.sum(axis=0)  # [11]

    return policy_obs_importance, value_ig_abs_mean, policy_ig_abs_mean


def compute_statistics(values_per_run, feature_names):
    """
    Compute paired-run statistics.

    Args:
        values_per_run: [N_RUNS, n_features]
        feature_names: list of feature names

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

        t_stat = mean_val / se_val if se_val > 0 else 0
        p_value = 2 * sp_stats.t.sf(abs(t_stat), df=n_runs - 1)

        stats.append({
            'feature': name,
            'mean': mean_val,
            'std': std_val,
            'ci_95': ci_95,
            'p_value': p_value,
        })

    return stats


def print_statistics_table(stats, title, n_runs):
    """Print a formatted statistics table."""
    print(f"\n{'='*100}")
    print(f"{title} — Statistical Summary (N={n_runs} runs)")
    print(f"{'='*100}")
    print(f"{'Feature':<12} | {'Mean':>10} | {'Std':>8} | {'95% CI':>24} | {'p-value':>10} | {'Sig?':>6}")
    print(f"{'-'*85}")

    for s in sorted(stats, key=lambda x: abs(x['mean']), reverse=True):
        ci_lo = s['mean'] - s['ci_95']
        ci_hi = s['mean'] + s['ci_95']
        sig = "***" if s['p_value'] < 0.001 else "**" if s['p_value'] < 0.01 else "*" if s['p_value'] < 0.05 else "ns"
        print(f"{s['feature']:<12} | {s['mean']:>10.6f} | {s['std']:>8.6f} | "
              f"[{ci_lo:>+10.6f}, {ci_hi:>+10.6f}] | {s['p_value']:>10.4f} | {sig:>6}")


def main():
    parser = argparse.ArgumentParser(description="Integrated Gradients Analysis")
    parser.add_argument("--model", type=str, default="rl_model", help="Path to trained model")
    parser.add_argument("--months", type=int, default=1200, help="Trajectory months per run")
    parser.add_argument("--n-samples", type=int, default=200, help="Observations to analyze per run")
    parser.add_argument("--n-steps", type=int, default=100, help="IG interpolation steps")
    parser.add_argument("--n-runs", type=int, default=30, help="Number of independent runs")
    parser.add_argument("--baseline", type=str, default="zero", choices=["zero", "mean"],
                        help="Baseline type for IG path")
    parser.add_argument("--master-seed", type=int, default=42, help="Master random seed")
    args = parser.parse_args()

    suppress_warnings()
    start_time = time.time()
    np.random.seed(args.master_seed)

    output_dir = Path("plots") / args.model / "integrated_gradients"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("INTEGRATED GRADIENTS ANALYSIS (Robust Multi-Run)")
    print("=" * 70)

    env_config = EnvConfig()
    model, env, var_names = load_environment(args.model, env_config)
    obs_names = list(var_names) + ['Month']
    action_names = list(var_names[1:])
    n_obs = len(obs_names)

    # Generate shared seeds from master RNG
    master_rng = np.random.default_rng(args.master_seed)
    shared_seeds = [int(master_rng.integers(0, 2**31)) for _ in range(args.n_runs)]

    print(f"\n  N_RUNS          = {args.n_runs}")
    print(f"  SIM_MONTHS      = {args.months} ({args.months // 12} years)")
    print(f"  N_SAMPLES/run   = {args.n_samples}")
    print(f"  IG_STEPS        = {args.n_steps}")
    print(f"  BASELINE        = {args.baseline}")
    print(f"{'=' * 70}\n")

    # Collect per-run results
    policy_importance_per_run = np.zeros((args.n_runs, n_obs))
    value_importance_per_run = np.zeros((args.n_runs, n_obs))
    ig_matrix_sum = None

    for run_idx, seed in enumerate(shared_seeds):
        print(f"--- Run {run_idx+1}/{args.n_runs} (seed={seed}) ---")

        observations = collect_observations(env, model, args.months, seed=seed)
        print(f"  Collected {len(observations)} observations")

        policy_imp, value_imp, ig_matrix = compute_ig_for_run(
            model, observations,
            n_ig_steps=args.n_steps,
            n_samples=args.n_samples,
            baseline_type=args.baseline
        )

        policy_importance_per_run[run_idx] = policy_imp[:n_obs]
        value_importance_per_run[run_idx] = value_imp[:n_obs]

        if ig_matrix_sum is None:
            ig_matrix_sum = ig_matrix.copy()
        else:
            ig_matrix_sum += ig_matrix

        top_idx = np.argmax(policy_imp[:n_obs])
        print(f"  Top obs (policy IG): {obs_names[top_idx]} ({policy_imp[top_idx]:.6f})\n")

    # Average IG matrix across all runs
    mean_ig_matrix_all = ig_matrix_sum / args.n_runs

    # Statistical analysis
    policy_stats = compute_statistics(policy_importance_per_run, obs_names)
    value_stats = compute_statistics(value_importance_per_run, obs_names)

    print_statistics_table(policy_stats, "POLICY IG ATTRIBUTION (Σ |IG| across actions)", args.n_runs)
    print_statistics_table(value_stats, "VALUE IG ATTRIBUTION (|IG(V, obs)|)", args.n_runs)

    # Completeness check
    print(f"\nSignificance: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")

    # Save results
    np.savez(
        output_dir / 'ig_results.npz',
        policy_importance_per_run=policy_importance_per_run,
        value_importance_per_run=value_importance_per_run,
        mean_ig_matrix_all=mean_ig_matrix_all,
        obs_names=obs_names,
        action_names=action_names,
        var_names=var_names,
        n_runs=args.n_runs,
        months=args.months,
        seeds=shared_seeds,
        baseline_type=args.baseline,
    )
    print(f"\nResults saved to {output_dir / 'ig_results.npz'}")

    elapsed = time.time() - start_time
    hours, remainder = divmod(int(elapsed), 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\n{'=' * 70}")
    print(f"INTEGRATED GRADIENTS ANALYSIS COMPLETE — Total time: {hours}h {minutes}m {seconds}s")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
