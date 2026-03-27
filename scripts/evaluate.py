"""
Evaluation script for trained ENSO RL agent.
- Basic for With RL vs without RL
- Trajectory for plotting traj
- intervention for ablation study results

Usage:
    uv run scripts/evaluate.py --model ppo_enso_model
    uv run scripts/evaluate.py --model model --intervention --months 2400
    uv run scripts/evaluate.py --model model --all --no-wandb
    uv run scripts/evaluate.py --model model --basic
    uv run scripts/evaluate.py --model model --intervention
    uv run scripts/evaluate.py --model model --trajectory
"""
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wandb
from pathlib import Path
from scipy import stats as sp_stats

# Add repo root to path for imports
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from stable_baselines3 import PPO

# Import configurations and utilities
from config import EnvConfig
from utils.data_processing import load_observational_data, prepare_xro_parameters
from utils.evaluation import evaluate_agent, simulate_trajectory
from utils.visualization import (
    plot_control_actions, plot_state_variables,
    plot_robust_interventional, plot_nino_classification,
    plot_action_kde_by_event, plot_state_kde_by_event
)
from utils.enso_table import (
    save_enso_table_html, save_enso_table_matplotlib, log_enso_table_wandb
)
from envs import XROMultiYearEnv
from utils import suppress_warnings
from XRO.core import XRO


def load_environment(model_path: str, env_config: EnvConfig):
    """
    Load trained model and create environment.
    
    Args:
        model_path (str): Path to saved model
        env_config (EnvConfig): Environment configuration
        
    Returns:
        tuple: (model, env, var_names)
    """
    print("\n" + "="*70)
    print("LOADING MODEL AND ENVIRONMENT")
    print("="*70)
    
    # Check model exists
    model_path_str = model_path
    
    # Add .zip extension if not present
    if not model_path_str.endswith('.zip'):
        model_path_str += '.zip'
    
    # Prepend models/ if not already present
    if not model_path_str.startswith('models'):
        model_path_str = f'models/{model_path_str}'
    
    model_path = Path(model_path_str)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}\n Make sure you have trained the model")
    
    # Load data
    print("\nLoading observational data...")
    obs_ds, train_ds, var_names, bounds = load_observational_data(
        env_config.data_config["data_path"],
        env_config.data_config["train_start"],
        env_config.data_config["train_end"]
    )
    
    # Prepare XRO parameters
    print("Preparing XRO model...")
    model_xro = XRO()
    params = prepare_xro_parameters(model_xro, train_ds, var_names, bounds)
    params['threshold'] = env_config.threshold
    
    # Create environment
    print("Creating environment...")
    env = XROMultiYearEnv(
        params=params,
        train_ds=train_ds,
        var_names=var_names,
        max_steps=env_config.max_steps
    )
    
    # Load model
    print(f"Loading PPO model from {model_path}...")
    model = PPO.load(str(model_path), env=env)
    
    print("[OK] Model and environment loaded successfully!")
    
    return model, env, var_names


def run_basic_evaluation(model, env, num_months=240, wandb_enabled=False):
    """
    Run basic evaluation comparing agent vs baseline.
    
    Args:
        model: Trained model
        env: Environment
        num_months (int): Number of evaluation months
        wandb_enabled (bool): Whether W&B is enabled
        
    Returns:
        dict: Evaluation results
    """
    print("\n" + "="*70)
    print("BASIC EVALUATION: AGENT VS BASELINE")
    print("="*70)
    
    print(f"\nEvaluating agent performance over {num_months} months...")
    prob_with_agent = evaluate_agent(env, agent=model, continuous_steps=num_months)
    
    print(f"Evaluating baseline (zero actions)...")
    prob_baseline = evaluate_agent(env, agent=None, continuous_steps=num_months)
    
    improvement = (prob_with_agent - prob_baseline) * 100
    ratio = prob_with_agent / prob_baseline if prob_baseline > 0 else 1.0
    
    print("\n" + "-"*70)
    print("Results:")
    print("-"*70)
    print(f"Multi-year events with agent:    {prob_with_agent:.2%}")
    print(f"Multi-year events (baseline):    {prob_baseline:.2%}")
    print(f"Improvement:                     {improvement:+.2f} percentage points")
    print(f"Improvement ratio:               {ratio:.2f}x")
    print("-"*70)
    
    results = {
        'evaluation/multi_year_events_with_agent': prob_with_agent,
        'evaluation/multi_year_events_without_agent': prob_baseline,
        'evaluation/improvement_percentage_points': improvement / 100,
        'evaluation/improvement_ratio': ratio,
    }
    
    # Log to W&B if enabled
    if wandb_enabled and wandb.run is not None:
        wandb.log(results)
    
    return results


def run_interventional_analysis(model, env, var_names, disable_idx=None, num_months=1200,
                                 n_runs=30, master_seed=42, wandb_enabled=False):
    """
    Run robust interventional analysis (ablation study) with N paired-seed trials.
    
    For each trial, baseline + all 9 interventions share the same seed, ensuring:
    - Fixed initial conditions (env.reset(seed=s) picks same starting state)
    - Fixed noise sequence (env's RNG produces identical XRO noise realizations)
    - Averaged over N runs — N paired ΔR samples per feature
    
    The only difference between conditions is the disabled action → clean causal signal.
    
    Args:
        model: Trained model
        env: Environment
        var_names (list): Variable names
        disable_idx (int or None): Index to disable (-1 for all)
        num_months (int): Number of simulation months per run
        n_runs (int): Number of paired trials
        master_seed (int): Master random seed for reproducibility
        wandb_enabled (bool): Whether W&B is enabled
        
    Returns:
        list: Delta R values with statistical results
    """
    print("\n" + "="*80)
    print("ROBUST INTERVENTIONAL ANALYSIS (ABLATION STUDY)")
    print("="*80)
    
    controllable_vars = list(var_names[1:])  # Skip Nino34
    n_features = len(controllable_vars)
    
    # Generate shared seeds from master RNG
    master_rng = np.random.default_rng(master_seed)
    shared_seeds = [int(master_rng.integers(0, 2**31)) for _ in range(n_runs)]
    
    print(f"  N_RUNS       = {n_runs}")
    print(f"  SIM_MONTHS   = {num_months} ({num_months//12} years per run)")
    print(f"  Features     = {n_features}")
    print(f"  Total sims   = {n_runs} x {n_features + 1} = {n_runs * (n_features + 1)}")
    print("="*80 + "\n")
    
    # Storage: rows = runs, cols = [baseline, feat_0, feat_1, ..., feat_8]
    all_rewards = np.zeros((n_runs, n_features + 1))
    all_mye_probs = np.zeros((n_runs, n_features + 1))
    
    for run_idx, seed in enumerate(shared_seeds):
        print(f"--- Run {run_idx+1}/{n_runs} (seed={seed}) ---")
        
        # Baseline (full control) with this seed
        baseline_sim = simulate_trajectory(
            env, agent=model, num_months=num_months,
            disable_control_for_idx=None, debug_mode=False, seed=seed
        )
        all_rewards[run_idx, 0] = baseline_sim['avg_reward']
        all_mye_probs[run_idx, 0] = baseline_sim['mye_probability']
        
        # Each intervention with the SAME seed → paired comparison
        for feat_idx in range(n_features):
            disabled_sim = simulate_trajectory(
                env, agent=model, num_months=num_months,
                disable_control_for_idx=feat_idx, debug_mode=False, seed=seed
            )
            all_rewards[run_idx, feat_idx + 1] = disabled_sim['avg_reward']
            all_mye_probs[run_idx, feat_idx + 1] = disabled_sim['mye_probability']
        
        # Quick summary for this run
        run_baseline = all_rewards[run_idx, 0]
        run_deltas = all_rewards[run_idx, 1:] - run_baseline
        worst_feat = controllable_vars[np.argmin(run_deltas)]
        print(f"  Baseline reward: {run_baseline:.4f} | Strongest driver: {worst_feat} (ΔR={run_deltas.min():.4f})\n")
    
    print(f"\n{'='*80}")
    print(f"All {n_runs} paired trials completed.")
    print(f"{'='*80}")
    
    # === Statistical Analysis ===
    # Compute paired ΔR: shape [n_runs, n_features]
    delta_r_matrix = all_rewards[:, 1:] - all_rewards[:, 0:1]
    
    mean_delta_r = delta_r_matrix.mean(axis=0)
    std_delta_r = delta_r_matrix.std(axis=0, ddof=1)
    se_delta_r = std_delta_r / np.sqrt(n_runs)
    ci_95 = sp_stats.t.ppf(0.975, df=n_runs - 1) * se_delta_r
    
    # One-sample t-test: is ΔR significantly different from 0?
    t_stats = mean_delta_r / se_delta_r
    p_values = 2 * sp_stats.t.sf(np.abs(t_stats), df=n_runs - 1)
    
    # MYE probability deltas
    delta_mye_matrix = all_mye_probs[:, 1:] - all_mye_probs[:, 0:1]
    mean_delta_mye = delta_mye_matrix.mean(axis=0)
    
    print(f"\n{'='*100}")
    print(f"PAIRED INTERVENTIONAL ANALYSIS — Statistical Summary (N={n_runs} runs)")
    print(f"{'='*100}")
    print(f"{'Feature':<10} | {'Mean ΔR':>10} | {'Std':>8} | {'95% CI':>20} | {'p-value':>10} | {'Sig?':>6} | {'ΔMYE%':>8}")
    print(f"{'-'*100}")
    
    for i, feat in enumerate(controllable_vars):
        ci_lo = mean_delta_r[i] - ci_95[i]
        ci_hi = mean_delta_r[i] + ci_95[i]
        sig = "***" if p_values[i] < 0.001 else "**" if p_values[i] < 0.01 else "*" if p_values[i] < 0.05 else "ns"
        print(f"{feat:<10} | {mean_delta_r[i]:>+10.4f} | {std_delta_r[i]:>8.4f} | [{ci_lo:>+8.4f}, {ci_hi:>+8.4f}] | {p_values[i]:>10.4f} | {sig:>6} | {mean_delta_mye[i]:>+7.2%}")
    
    print(f"\nBaseline avg reward (mean over {n_runs} runs): {all_rewards[:, 0].mean():.4f} +/- {all_rewards[:, 0].std(ddof=1):.4f}")
    print(f"Significance: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
    
    # Build delta_r_values list (used by downstream code)
    delta_r_values = []
    for i, feat in enumerate(controllable_vars):
        delta_r_values.append({
            'feature': feat,
            'delta_r': mean_delta_r[i],
            'ci_95': ci_95[i],
            'std': std_delta_r[i],
            'p_value': p_values[i],
            'delta_mye': mean_delta_mye[i],
            'avg_reward': all_rewards[:, i + 1].mean(),
        })
    
    # Ranking
    print(f"\n{'='*80}")
    print("Delta R Ranking (Most Negative = Strongest Driver):")
    print(f"{'='*80}")
    print(f"Baseline Reward (mean +/- std): {all_rewards[:, 0].mean():.4f} +/- {all_rewards[:, 0].std(ddof=1):.4f}\n")
    
    for item in sorted(delta_r_values, key=lambda x: x['delta_r']):
        sig = "***" if item['p_value'] < 0.001 else "**" if item['p_value'] < 0.01 else "*" if item['p_value'] < 0.05 else "ns"
        direction = "DRIVER" if item['delta_r'] < 0 and item['p_value'] < 0.05 else "HELPS" if item['delta_r'] > 0 and item['p_value'] < 0.05 else "UNCLEAR"
        print(f"{item['feature']:<10} | ΔR: {item['delta_r']:>+8.4f} +/- {item['ci_95']:.4f} | {direction} {sig}")
    
    # === Robust Plot (matching notebook) ===
    print("\nGenerating robust interventional analysis plot...")
    plot_robust_interventional(delta_r_values, n_runs, wandb_enabled)
    print("[OK] Robust plot saved to plots/interventional_analysis_robust.png")
    
    # Save numerical results
    output_dir = Path("plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / 'interventional_results.npz',
        all_rewards=all_rewards,
        all_mye_probs=all_mye_probs,
        delta_r_matrix=delta_r_matrix,
        delta_mye_matrix=delta_mye_matrix,
        controllable_vars=controllable_vars,
        n_runs=n_runs,
        num_months=num_months,
        seeds=shared_seeds,
    )
    print(f"[OK] Results saved to {output_dir / 'interventional_results.npz'}")
    
    return delta_r_values


def run_trajectory_analysis(model, env, var_names, num_months=240, threshold=0.5, wandb_enabled=False):
    """
    Run trajectory analysis and visualization.
    
    Args:
        model: Trained model
        env: Environment
        var_names (list): Variable names
        num_months (int): Duration of simulation
        threshold (float): ENSO event threshold from env_config
        wandb_enabled (bool): Whether W&B is enabled
        
    Returns:
        dict: Simulation data and trajectory dataframe
    """
    print("\n" + "="*70)
    print("TRAJECTORY ANALYSIS")
    print("="*70)
    
    print(f"\nSimulating {num_months} months ({num_months/12:.1f} years) of continuous control...")
    
    sim = simulate_trajectory(
        env, 
        agent=model, 
        num_months=num_months,
        debug_mode=True
    )
    
    print(f"\nSimulation Results:")
    print("-"*70)
    print(f"Duration:                 {sim['no_months']} months ({sim['no_months']/12:.1f} years)")
    print(f"Classified event:         {sim['classified_event']}")
    print(f"Average reward:           {sim['avg_reward']:.4f}")
    print(f"Multi-year event prob:    {sim['mye_probability']:.2%}")
    print(f"Elapsed time:             {sim['elapsed_time']:.2f}s")
    print("-"*70)
    
    # Summary statistics
    enso_traj = sim['enso_traj']
    print(f"\nENSO Index Statistics:")
    print("-"*70)
    print(f"Mean:                     {np.mean(enso_traj):.4f}")
    print(f"Std dev:                  {np.std(enso_traj):.4f}")
    print(f"Min:                      {np.min(enso_traj):.4f}")
    print(f"Max:                      {np.max(enso_traj):.4f}")
    print("-"*70)
    
    # Save trajectory data
    trajectory_df = pd.DataFrame({
        'enso_index': sim['enso_traj'][:-1], # remove last to match len(action)
    })
    
    # Add actions
    for i, var_name in enumerate(var_names[1:]):
        if i < sim['actions_traj'].shape[1]:
            trajectory_df[f'action_{var_name}'] = sim['actions_traj'][:, i]
    
    # Add states
    for i, var_name in enumerate(var_names):
        if i < sim['states_traj'].shape[1]:
            trajectory_df[f'state_{var_name}'] = sim['states_traj'][:-1, i] # remove last to match len(action)
    
    # Save to CSV
    output_file = "trajectory_analysis.csv"
    trajectory_df.to_csv(output_file, index=False)
    print(f"\n[OK] Trajectory data saved to {output_file}")
    
    # Generate plots
    print("\nGenerating visualization plots...")
    plot_control_actions(sim['actions_traj'], var_names, num_months=num_months, wandb_enabled=wandb_enabled)
    plot_state_variables(sim['states_traj'][:-1], var_names, threshold=threshold, num_months=num_months, wandb_enabled=wandb_enabled)
    plot_nino_classification(sim['enso_traj'], sim['classified_event_array'], threshold=threshold, num_months=num_months, wandb_enabled=wandb_enabled)
    
    # Generate KDE plots by event type
    print("Generating KDE plots by event type...")
    plot_action_kde_by_event(sim['actions_traj'], sim['classified_event_array'], var_names, wandb_enabled=wandb_enabled)
    plot_state_kde_by_event(sim['states_traj'][:-1], sim['classified_event_array'], var_names, wandb_enabled=wandb_enabled)
    
    print("[OK] Plots saved to plots/ folder and logged to W&B")
    
    # Generate ENSO table with colors
    print("\nGenerating ENSO index table...")
    save_enso_table_html(sim['enso_traj'], output_path="plots/enso_table.html", threshold=threshold, wandb_enabled=wandb_enabled)
    save_enso_table_matplotlib(sim['enso_traj'], output_path="plots/enso_table.png", threshold=threshold)
    if wandb_enabled:
        log_enso_table_wandb(sim['enso_traj'], threshold=threshold)
    
    print("[OK] ENSO table saved as HTML and PNG")
    print("     → Open plots/enso_table.html in your browser for interactive table")
    print("     → View plots/enso_table.png for image version")
    
    return {'sim': sim, 'trajectory_df': trajectory_df}


def main():
    """Main evaluation pipeline."""
    
    parser = argparse.ArgumentParser(description="Evaluate ENSO RL Agent")
    parser.add_argument("--model", type=str, default="rl_model", help="Path to trained model")
    parser.add_argument("--basic", action="store_true", help="Run basic evaluation")
    parser.add_argument("--intervention", action="store_true", help="Run interventional analysis")
    parser.add_argument("--trajectory", action="store_true", help="Run trajectory analysis")
    parser.add_argument("--all", action="store_true", help="Run all evaluations")
    parser.add_argument("--months", type=int, default=1200, help="Simulation months per run")
    parser.add_argument("--n-runs", type=int, default=30, help="Number of paired trials for interventional analysis")
    parser.add_argument("--master-seed", type=int, default=42, help="Master random seed")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    args = parser.parse_args()
    
    # Set to run all if none specified
    if not (args.basic or args.intervention or args.trajectory):
        args.all = True
    
    print("\n")
    print("-"*20 + "ENSO RL AGENT EVALUATION PIPELINE" + "-"*20)
    
    try:
        suppress_warnings()
        
        # Initialize W&B flag early (before any code that might fail)
        wandb_enabled = not args.no_wandb
        
        # Load environment
        env_config = EnvConfig()
        model, env, var_names = load_environment(args.model, env_config)
        
        # Initialize W&B (enabled by default)
        wandb_enabled = not args.no_wandb
        if wandb_enabled:
            from config import WandbConfig
            from datetime import datetime
            wandb_config = WandbConfig()
            run_name = datetime.now().strftime(r"%H:%M %d-%m-%y")
            wandb.init(
                project=wandb_config.project,
                entity=wandb_config.entity,
                name=run_name,
                tags=["evaluation", "enso", "ppo"],
                group="eval",
                job_type="evaluation",
            )
            wandb.config.update({
                "model_path": args.model,
                "months": args.months,
                "n_runs": args.n_runs,
                "master_seed": args.master_seed,
            })
            print(f"[OK] W&B initialized — {wandb.run.url}")
        else:
            print("[INFO] W&B logging disabled (--no-wandb)")
        
        # Run evaluations
        if args.all or args.basic:
            run_basic_evaluation(model, env, num_months=args.months, wandb_enabled=wandb_enabled)

        if args.all or args.intervention:
            run_interventional_analysis(model, env, var_names, num_months=args.months,
                                        n_runs=args.n_runs, master_seed=args.master_seed,
                                        wandb_enabled=wandb_enabled)
        
        if args.all or args.trajectory:
            run_trajectory_analysis(model, env, var_names, num_months=args.months, threshold=env_config.threshold, wandb_enabled=wandb_enabled)
        
        # Finish W&B run
        if wandb_enabled and wandb.run is not None:
            wandb.finish()
        
        print("-"*20 + "EVALUATION PIPELINE COMPLETED" + "-"*20)
        
    except Exception as e:
        print(f"\n[ERROR] Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        if wandb_enabled and wandb.run is not None:
            wandb.finish(exit_code=1)


if __name__ == "__main__":
    main()
