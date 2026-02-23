"""
Evaluation script for trained ENSO RL agent.

Usage:
    python scripts/evaluate.py --model ppo_enso_model.zip
    python scripts/evaluate.py --model model.zip --intervention --disable-idx 0
    python scripts/evaluate.py --model model.zip --trajectory --months 2400
"""
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wandb
from pathlib import Path

# Add repo root to path for imports
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from stable_baselines3 import PPO

# Import configurations and utilities
from config import EnvConfig
from utils.data_processing import load_observational_data, prepare_xro_parameters
from utils.evaluation import evaluate_agent, simulate_trajectory
from utils.visualization import (
    plot_enso_trajectory, plot_control_actions, plot_state_variables,
    plot_feature_importance, plot_event_classification, plot_action_kde_by_event, 
    plot_state_kde_by_event
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


def run_interventional_analysis(model, env, var_names, disable_idx=None, num_months=240, wandb_enabled=False):
    """
    Run interventional analysis (ablation study).
    
    Args:
        model: Trained model
        env: Environment
        var_names (list): Variable names
        disable_idx (int or None): Index to disable (-1 for all)
        num_months (int): Number of simulation months
        wandb_enabled (bool): Whether W&B is enabled
        
    Returns:
        list: Delta R values with plotting
    """
    print("\n" + "="*70)
    print("INTERVENTIONAL ANALYSIS (ABLATION STUDY)")
    print("="*70)
    
    # Get baseline
    print("\nCollecting baseline performance...")
    baseline_sim = simulate_trajectory(
        env, agent=model, num_months=num_months,
        disable_control_for_idx=None, debug_mode=False
    )
    baseline_reward = baseline_sim['avg_reward']
    baseline_prob = baseline_sim['mye_probability']
    
    print(f"Baseline reward:             {baseline_reward:.4f}")
    print(f"Baseline multi-year prob:    {baseline_prob:.2%}")
    
    # Test disabling each action
    controllable_vars = var_names[1:]  # Skip Nino34
    
    print(f"\nTesting impact of disabling each action ({len(controllable_vars)} variables)...")
    print("-"*70)
    print(f"{'Variable':<12} | {'Avg Reward':<12} | {'Delta R':<12} | {'MYE Prob':<12}")
    print("-"*70)
    
    delta_r_values = []
    
    for i, var_name in enumerate(controllable_vars):
        sim = simulate_trajectory(
            env, agent=model, num_months=num_months,
            disable_control_for_idx=i, debug_mode=False
        )
        
        delta_r = sim['avg_reward'] - baseline_reward
        delta_r_values.append({
            'feature': var_name,
            'avg_reward': sim['avg_reward'],
            'delta_r': delta_r,
            'mye_probability': sim['mye_probability'],
        })
        
        significance = "CRITICAL" if delta_r < -0.01 else "POSITIVE" if delta_r > 0.01 else "NEUTRAL"
        print(f"{var_name:<12} | {sim['avg_reward']:<12.4f} | {delta_r:<12.4f} | {sim['mye_probability']:<12.2%}")
    
    print("-"*70)
    
    # Sort by importance
    sorted_results = sorted(delta_r_values, key=lambda x: x['delta_r'])
    print("\nRanking by importance (most to least important):")
    print("-"*70)
    for i, result in enumerate(sorted_results, 1):
        print(f"{i}. {result['feature']:<10} - DR: {result['delta_r']:>8.4f}")
    
    # Generate feature importance plot
    print("\nGenerating feature importance plot...")
    plot_feature_importance(delta_r_values, wandb_enabled=wandb_enabled)
    print("[OK] Feature importance plot saved to plots/04_feature_importance.png")
    
    return delta_r_values


def run_trajectory_analysis(model, env, var_names, num_months=240, wandb_enabled=False):
    """
    Run trajectory analysis and visualization.
    
    Args:
        model: Trained model
        env: Environment
        var_names (list): Variable names
        num_months (int): Duration of simulation
        wandb_enabled (bool): Whether W&B is enabled
        
    Returns:
        dict: Simulation data and trajectory dataframe
    """
    print("\n" + "="*70)
    print("TRAJECTORY ANALYSIS")
    print("="*70)
    
    print(f"\nSimulating {num_months} months ({num_months/12:.1f} years) of continuous control...")
    
    sim = simulate_trajectory(
        env, agent=model, num_months=num_months,
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
    plot_enso_trajectory(sim['enso_traj'], threshold=1.0, num_months=num_months, wandb_enabled=wandb_enabled)
    plot_control_actions(sim['actions_traj'], var_names, num_months=num_months, wandb_enabled=wandb_enabled)
    plot_state_variables(sim['states_traj'][:-1], var_names, threshold=1.0, num_months=num_months, wandb_enabled=wandb_enabled)
    plot_event_classification(sim['enso_traj'], sim['classified_event_array'], num_months=num_months, wandb_enabled=wandb_enabled)
    
    # Generate KDE plots by event type
    print("Generating KDE plots by event type...")
    plot_action_kde_by_event(sim['actions_traj'], sim['classified_event_array'], var_names, wandb_enabled=wandb_enabled)
    plot_state_kde_by_event(sim['states_traj'][:-1], sim['classified_event_array'], var_names, wandb_enabled=wandb_enabled)
    
    print("[OK] Plots saved to plots/ folder and logged to W&B")
    
    # Generate ENSO table with colors
    print("\nGenerating ENSO index table...")
    save_enso_table_html(sim['enso_traj'], output_path="plots/enso_table.html", threshold=1.0, wandb_enabled=wandb_enabled)
    save_enso_table_matplotlib(sim['enso_traj'], output_path="plots/enso_table.png", threshold=1.0, wandb_enabled=wandb_enabled)
    if wandb_enabled:
        log_enso_table_wandb(sim['enso_traj'], threshold=1.0)
    
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
    parser.add_argument("--months", type=int, default=240, help="Simulation months")
    args = parser.parse_args()
    
    # Set to run all if none specified
    if not (args.basic or args.intervention or args.trajectory):
        args.all = True
    
    print("\n")
    print("-"*20 + "ENSO RL AGENT EVALUATION PIPELINE" + "-"*20)
    
    try:
        suppress_warnings()
        # Load environment
        env_config = EnvConfig()
        model, env, var_names = load_environment(args.model, env_config)
        
        # Check if W&B is enabled
        wandb_enabled = wandb.run is not None
        
        # Run evaluations
        if args.all or args.basic:
            run_basic_evaluation(model, env, num_months=args.months, wandb_enabled=wandb_enabled)

        if args.all or args.intervention:
            run_interventional_analysis(model, env, var_names, num_months=args.months, wandb_enabled=wandb_enabled)
        
        if args.all or args.trajectory:
            run_trajectory_analysis(model, env, var_names, num_months=args.months, wandb_enabled=wandb_enabled)
        
        print("-"*20 + "EVALUATION PIPELINE COMPLETED" + "-"*20)
        
    except Exception as e:
        print(f"\n[ERROR] Evaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
