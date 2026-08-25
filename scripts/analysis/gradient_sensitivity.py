"""
Gradient-Based Sensitivity Analysis for ENSO RL Agent.

Computes ∂action/∂observation (policy Jacobian) and ∂V/∂observation (value
function sensitivity) to determine which state variables the agent attends
to when deciding actions, and which observation dimensions most affect value.

Uses N independent paired-seed runs for statistical robustness with t-tests,
95% CIs, and significance testing.

Usage:
    python scripts/gradient_sensitivity.py --model rl_model
    python scripts/gradient_sensitivity.py --model rl_model --n-runs 30 --months 1200
"""
import sys
import time
import argparse
import numpy as np
import torch
from pathlib import Path
from scipy import stats as sp_stats

repo_root = Path(__file__).parent.parent
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
    params = prepare_xro_parameters(model_xro, train_ds, var_names,
                                    config=env_config)
    params['threshold'] = env_config.threshold

    env = XROMultiYearEnv(
        params=params, train_ds=train_ds,
        var_names=var_names, max_steps=env_config.max_steps
    )

    model = PPO.load(str(model_path), env=env)
    return model, env, var_names


def collect_observations(env, model, num_months, seed):
    """Collect observations from a trajectory for gradient analysis."""
    obs, _ = env.reset(seed=seed)
    observations = [obs.copy()]

    for step in range(num_months):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        observations.append(obs.copy())

    return np.array(observations)


def compute_policy_jacobian(model, observations, n_samples=None):
    """
    Compute ∂action/∂observation for the policy network.

    For each sampled observation, compute the full Jacobian matrix
    (9 actions × 11 obs dimensions) via backpropagation.

    Args:
        model: Trained PPO model
        observations: Array of observations [T, 11]
        n_samples: Number of observations to sample (None = all)

    Returns:
        np.ndarray: Mean absolute Jacobian [9, 11]
        np.ndarray: All Jacobians [n_samples, 9, 11]
    """
    policy = model.policy
    policy.eval()

    if n_samples is not None and n_samples < len(observations):
        indices = np.random.choice(len(observations), n_samples, replace=False)
        obs_subset = observations[indices]
    else:
        obs_subset = observations

    all_jacobians = []

    for obs in obs_subset:
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).requires_grad_(True)

        features = policy.extract_features(obs_tensor, policy.pi_features_extractor)
        latent_pi = policy.mlp_extractor.forward_actor(features)
        action_mean = policy.action_net(latent_pi)

        jacobian = torch.zeros(action_mean.shape[1], obs_tensor.shape[1])
        for j in range(action_mean.shape[1]):
            if obs_tensor.grad is not None:
                obs_tensor.grad.zero_()
            action_mean[0, j].backward(retain_graph=True)
            if obs_tensor.grad is not None:
                jacobian[j] = obs_tensor.grad[0].clone()

        all_jacobians.append(jacobian.detach().numpy())

    all_jacobians = np.array(all_jacobians)
    mean_abs_jacobian = np.mean(np.abs(all_jacobians), axis=0)

    return mean_abs_jacobian, all_jacobians


def compute_value_gradient(model, observations, n_samples=None):
    """
    Compute ∂V/∂observation for the value function.

    Args:
        model: Trained PPO model
        observations: Array of observations [T, 11]
        n_samples: Number of observations to sample

    Returns:
        np.ndarray: Mean absolute value gradient [11]
        np.ndarray: All gradients [n_samples, 11]
    """
    policy = model.policy
    policy.eval()

    if n_samples is not None and n_samples < len(observations):
        indices = np.random.choice(len(observations), n_samples, replace=False)
        obs_subset = observations[indices]
    else:
        obs_subset = observations

    all_grads = []

    for obs in obs_subset:
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).requires_grad_(True)

        features = policy.extract_features(obs_tensor, policy.vf_features_extractor)
        latent_vf = policy.mlp_extractor.forward_critic(features)
        value = policy.value_net(latent_vf)

        value.backward()
        if obs_tensor.grad is not None:
            all_grads.append(obs_tensor.grad[0].clone().detach().numpy())

    all_grads = np.array(all_grads)
    mean_abs_grad = np.mean(np.abs(all_grads), axis=0)

    return mean_abs_grad, all_grads


def compute_run_importance(env, model, num_months, n_samples, seed):
    """
    Compute observation importance for a single run (trajectory).

    Args:
        env: Environment
        model: Trained PPO model
        num_months: Months to simulate
        n_samples: Observations to sample for gradient computation
        seed: Random seed for env.reset()

    Returns:
        policy_obs_importance: [11] sum of |∂action/∂obs| across actions
        value_obs_importance: [11] mean |∂V/∂obs|
        mean_jacobian: [9, 11] mean absolute Jacobian
    """
    observations = collect_observations(env, model, num_months, seed=seed)

    mean_jacobian, _ = compute_policy_jacobian(model, observations, n_samples=n_samples)
    mean_vgrad, _ = compute_value_gradient(model, observations, n_samples=n_samples)

    # Policy obs importance = sum |∂action/∂obs| across all 9 actions
    policy_obs_importance = mean_jacobian.sum(axis=0)

    return policy_obs_importance, mean_vgrad, mean_jacobian


def compute_statistics(values_per_run, feature_names):
    """
    Compute paired-run statistics.

    Args:
        values_per_run: [N_RUNS, n_features] per-run importance
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

        # t-test: is importance significantly different from 0?
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
    parser = argparse.ArgumentParser(description="Gradient-Based Sensitivity Analysis")
    parser.add_argument("--model", type=str, default="rl_model", help="Path to trained model")
    parser.add_argument("--months", type=int, default=1200, help="Trajectory months per run")
    parser.add_argument("--n-samples", type=int, default=500, help="Observations to sample per run")
    parser.add_argument("--n-runs", type=int, default=30, help="Number of independent runs")
    parser.add_argument("--master-seed", type=int, default=42, help="Master random seed")
    args = parser.parse_args()

    suppress_warnings()
    np.random.seed(args.master_seed)

    output_dir = Path("plots") / args.model / "gradient_sensitivity"
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    print("=" * 70)
    print("GRADIENT-BASED SENSITIVITY ANALYSIS (Robust Multi-Run)")
    print("=" * 70)

    env_config = EnvConfig()
    model, env, var_names = load_environment(args.model, env_config)
    action_names = list(var_names[1:])
    obs_names = list(var_names) + ['Month']
    n_obs = len(obs_names)

    # Generate shared seeds from master RNG
    master_rng = np.random.default_rng(args.master_seed)
    shared_seeds = [int(master_rng.integers(0, 2**31)) for _ in range(args.n_runs)]

    print(f"\n  N_RUNS          = {args.n_runs}")
    print(f"  SIM_MONTHS      = {args.months} ({args.months // 12} years)")
    print(f"  N_SAMPLES/run   = {args.n_samples}")
    print(f"{'=' * 70}\n")

    # Collect per-run results
    policy_importance_per_run = np.zeros((args.n_runs, n_obs))
    value_importance_per_run = np.zeros((args.n_runs, n_obs))
    jacobian_sum = None

    for run_idx, seed in enumerate(shared_seeds):
        print(f"--- Run {run_idx+1}/{args.n_runs} (seed={seed}) ---")

        policy_imp, value_imp, mean_jac = compute_run_importance(
            env, model, args.months, args.n_samples, seed
        )

        policy_importance_per_run[run_idx] = policy_imp[:n_obs]
        value_importance_per_run[run_idx] = value_imp[:n_obs]

        if jacobian_sum is None:
            jacobian_sum = mean_jac.copy()
        else:
            jacobian_sum += mean_jac

        top_idx = np.argmax(policy_imp[:n_obs])
        print(f"  Top obs (policy): {obs_names[top_idx]} ({policy_imp[top_idx]:.4f})\n")

    # Average Jacobian across all runs
    mean_jacobian_all = jacobian_sum / args.n_runs

    # Statistical analysis
    policy_stats = compute_statistics(policy_importance_per_run, obs_names)
    value_stats = compute_statistics(value_importance_per_run, obs_names)

    print_statistics_table(policy_stats, "POLICY SENSITIVITY (Σ |∂action/∂obs|)", args.n_runs)
    print_statistics_table(value_stats, "VALUE SENSITIVITY (|∂V/∂obs|)", args.n_runs)

    print(f"\nSignificance: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")

    # Save results
    np.savez(
        output_dir / 'gradient_results.npz',
        policy_importance_per_run=policy_importance_per_run,
        value_importance_per_run=value_importance_per_run,
        mean_jacobian_all=mean_jacobian_all,
        obs_names=obs_names,
        action_names=action_names,
        var_names=var_names,
        n_runs=args.n_runs,
        months=args.months,
        seeds=shared_seeds,
    )
    print(f"\nResults saved to {output_dir / 'gradient_results.npz'}")

    elapsed = time.time() - start_time
    hours, remainder = divmod(int(elapsed), 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"\n{'=' * 70}")
    print(f"GRADIENT SENSITIVITY ANALYSIS COMPLETE — Total time: {hours}h {minutes}m {seconds}s")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
