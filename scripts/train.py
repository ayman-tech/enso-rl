"""
Main training script for ENSO RL agent.

Usage (normal and Full customizable run):
    uv run scripts/train.py --epochs 1000 --name "ppo-train"
    uv run scripts/train.py --epochs 1000 --name "ppo-train" --lr 0.0001 --no-wandb --debug
"""
import sys
import io
import time
import argparse
import warnings
from pathlib import Path

# Add repo root to path for imports
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

import wandb
import torch as th
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

# Import configurations and utilities
from config import TrainConfig, EnvConfig, WandbConfig
from utils import suppress_warnings, timer
from utils.seeding import SeedBundle, resolve_seeds, model_name
from utils.data_processing import (
    load_observational_data, 
    prepare_xro_parameters
)
from utils.evaluation import evaluate_agent
from callbacks import WandbCallback, TrainingHistoryCallback
from envs import XROMultiYearEnv
from XRO.core import XRO


def setup_environment(config: EnvConfig):
    """
    Set up and prepare the environment.
    
    Args:
        config (EnvConfig): Environment configuration
        
    Returns:
        tuple: (obs_ds, train_ds, var_names, bounds, params)
    """
    print("\n" + "="*50)
    print("SETTING UP ENVIRONMENT")
    print("="*50)
    
    # Suppress warnings
    suppress_warnings()
    
    # Load data
    print("\n1. Loading observational data...")
    obs_ds, train_ds, var_names, bounds = load_observational_data(
        config.data_config["data_path"],
        config.data_config["train_start"],
        config.data_config["train_end"]
    )
    print(f"   [OK] Loaded {len(var_names)} variables:")
    for var in var_names:
        print(f"     - {var}")
    
    # Prepare XRO parameters
    print("\n2. Preparing XRO model...")
    model = XRO()
    params = prepare_xro_parameters(model, train_ds, var_names, bounds)
    params['threshold'] = config.threshold
    print("   [OK] XRO model fitted and parameters extracted")
    
    return obs_ds, train_ds, var_names, bounds, params


def initialize_wandb(wandb_config: WandbConfig, train_config: TrainConfig, env_config: EnvConfig,
                     seeds: SeedBundle = None):
    """
    Initialize Weights & Biases tracking.

    Args:
        wandb_config (WandbConfig): W&B configuration
        train_config (TrainConfig): Training configuration
        env_config (EnvConfig): Environment configuration
        seeds (SeedBundle): Resolved per-axis seeds, logged for provenance.
    """
    print("\n3. Initializing Weights & Biases...")
    
    wandb.init(**wandb_config.to_dict())
    
    # Log hyperparameters
    hyperparameters = {
        "algorithm": "PPO",
        "policy": "MlpPolicy",
        "train_epochs": train_config.train_epochs,
        "env_max_steps": train_config.episode_length,
        "action_space_dim": env_config.action_dim,
        "observation_space_dim": env_config.obs_dim,
        "action_scale": env_config.action_scale,
        "enso_threshold": env_config.threshold,
        "debug_mode": train_config.debug_mode,
        "learning_rate": train_config.learning_rate,
        "gamma": train_config.gamma,
        "n_steps": train_config.n_steps,
    }
    if seeds is not None:
        hyperparameters.update(seeds.as_log_dict())

    wandb.config.update(hyperparameters)
    print(f"\t[OK] W&B initialized")
    print(f"\t[OK] View experiment at: {wandb.run.url}")


@timer
def train_ppo_agent(env, train_config: TrainConfig, wandb_config: WandbConfig,
                    seeds: SeedBundle, eval_env=None):
    """
    Train PPO agent on the environment.

    Args:
        env: Gymnasium environment
        train_config (TrainConfig): Training configuration
        wandb_config (WandbConfig): W&B configuration
        seeds (SeedBundle): Resolved per-axis seeds. `weight` seeds policy weight
            initialization (axis #1); `action` seeds the stochastic Gaussian
            exploration sampled from the policy via the PyTorch RNG (axis #2);
            `shuffle` seeds PPO's global-NumPy mini-batch shuffle (axis #3). The
            env-side axes (#4 start state, #5 physics noise) are seeded on the env
            objects themselves.
        eval_env: Separate env for periodic mye_prob evaluation (not the training
            env, whose rollout would be corrupted by eval resets). If None, the
            mye_prob eval callback is skipped.

    Returns:
        PPO: Trained model
    """
    print("\n" + "="*70)
    print("STARTING PPO TRAINING")
    print("="*70)
    print(f"Total timesteps: {round(train_config.train_months/12):,} yrs or {train_config.train_months:,} months")
    print(f"Learning rate: {train_config.learning_rate}")
    print(f"Discount factor (gamma): {train_config.gamma}")
    print(f"Update frequency (n_steps): {train_config.n_steps}")

    # Capture verbose output
    output_buffer = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = output_buffer

    try:
        # Create model. PPO(seed=...) seeds the PyTorch RNG used for weight init
        # (axis #1) at construction.
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=train_config.learning_rate,
            n_steps=train_config.n_steps,
            gamma=train_config.gamma,
            seed=seeds.weight,
            verbose=1
        )

        # Re-seed the two remaining optimization-side axes just before learning so
        # they are independent of weight init: the PyTorch RNG now drives action
        # sampling (axis #2) and the global NumPy RNG drives the mini-batch shuffle
        # (axis #3). The physics global-RNG isolation in xro_step keeps #3 clean.
        if seeds.action is not None:
            th.manual_seed(seeds.action)
        if seeds.shuffle is not None:
            np.random.seed(seeds.shuffle)

        # Train with W&B callback (+ periodic mye_prob evaluation if eval_env given)
        callbacks = [WandbCallback(verbose=1)]
        if eval_env is not None:
            # Evaluate roughly 20 times over the run (at least every 24k steps)
            eval_freq = max(train_config.n_steps,
                            min(24000, train_config.train_months // 20))
            name = wandb_config.name
            master = seeds.weight
            suffix = f"_seed{master}" if master is not None else None
            if (not seeds.has_override and suffix
                    and name.endswith(suffix) and len(name) > len(suffix)):
                out_dir = Path("plots") / name[:-len(suffix)]
                run_stem = f"train_seed{master}"
            else:
                out_dir = Path("plots") / name
                run_stem = "training"
            callbacks.append(TrainingHistoryCallback(
                eval_env=eval_env,
                out_dir=out_dir,
                run_stem=run_stem,
                model_name=name,
                eval_freq=eval_freq,
                eval_steps=train_config.sim_months,
                verbose=1,
            ))
        model.learn(
            total_timesteps=train_config.train_months,
            callback=callbacks
        )
        
        # Restore stdout
        sys.stdout = old_stdout
        verbose_output = output_buffer.getvalue()
        
        # Parse metrics
        print("[OK] Training completed successfully!")
        
        # Save model
        model.save(train_config.model_save_path)
        train_config.model_save_path = "models/"+wandb_config.name
        print(f"[OK] Model saved to {train_config.model_save_path}.zip")
        
        # Log model as artifact (only if W&B is enabled)
        if wandb.run is not None:
            artifact = wandb.Artifact(
                name=f"{wandb_config.name}-{wandb.run.id}",
                type="model",
                description="Trained PPO model for ENSO climate control"
            )
            artifact.add_file(f"{train_config.model_save_path}.zip")
            wandb.log_artifact(artifact)
            print("[OK] Model logged to W&B artifacts")
        
        return model
        
    except Exception as e:
        sys.stdout = old_stdout
        print(f"[ERROR] Training failed: {e}")
        raise


def evaluate_trained_model(env, model, eval_steps=6000):
    """
    Evaluate the trained model.
    
    Args:
        env: Gymnasium environment
        model: Trained model
        eval_steps (int): Number of steps to evaluate
        
    Returns:
        dict: Evaluation results
    """
    print("\n" + "="*70)
    print("EVALUATING TRAINED MODEL")
    print("="*70)
    
    print(f"\nEvaluating with agent over {eval_steps} continuous steps...")
    prob_with_control = evaluate_agent(env, agent=model, continuous_steps=eval_steps)
    
    print(f"Evaluating baseline (no control)...")
    prob_without_control = evaluate_agent(env, agent=None, continuous_steps=eval_steps)
    
    improvement = (prob_with_control - prob_without_control) * 100
    
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"With RL Agent Control:     {prob_with_control:.1%}")
    print(f"Without Control (Baseline): {prob_without_control:.1%}")
    print(f"Improvement: {improvement:+.1f} percentage points")
    print("="*70)
    
    # Log to W&B (only if initialized)
    evaluation_results = {
        "evaluation/multi_year_events_with_agent": prob_with_control,
        "evaluation/multi_year_events_without_agent": prob_without_control,
        "evaluation/improvement_percentage_points": improvement,
        "evaluation/improvement_ratio": prob_with_control / prob_without_control if prob_without_control > 0 else 1.0,
    }
    if wandb.run is not None:
        wandb.log(evaluation_results)
        print("\n[OK] Evaluation results logged to W&B!")
    
    return evaluation_results


def main():
    """Main training pipeline."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train ENSO RL Agent")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--seed", type=int, default=None,
                        help="Master random seed. By default every randomness axis "
                             "(weight/action/shuffle/init/physics) uses this value.")
    # Per-axis seed overrides for the randomness-sensitivity study. Override one axis
    # while the rest stay pinned at --seed to attribute outcome variance to that axis.
    # Requires --seed to be set.
    parser.add_argument("--seed-weight", type=int, default=None, help="Override seed: policy weight init")
    parser.add_argument("--seed-action", type=int, default=None, help="Override seed: Gaussian exploration")
    parser.add_argument("--seed-batch", type=int, default=None, help="Override seed: mini-batch shuffle")
    parser.add_argument("--seed-init", type=int, default=None, help="Override seed: env start state")
    parser.add_argument("--seed-physics", type=int, default=None, help="Override seed: XRO climate noise")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--name", type=str, default="train-run", help="Name of WandB training Run")
    args = parser.parse_args()
    
    # Load configurations
    train_config = TrainConfig()
    env_config = EnvConfig()
    wandb_config = WandbConfig()
    
    # Override with command line args
    if args.debug:
        train_config.debug_mode = True
    if args.epochs:
        train_config.train_epochs = args.epochs
        train_config.train_months = args.epochs * train_config.n_steps
    if args.lr:
        train_config.learning_rate = args.lr
    if args.no_wandb:
        wandb_config.mode = "disabled"

    # Resolve the five randomness-axis seeds. By default every axis uses --seed;
    # per-axis overrides (raises if given without --seed) drive the sensitivity study.
    seeds = resolve_seeds(args.seed, {
        "weight": args.seed_weight,
        "action": args.seed_action,
        "shuffle": args.seed_batch,
        "init": args.seed_init,
        "physics": args.seed_physics,
    })
    train_config.seed = seeds.weight  # kept for any downstream reference/logging

    # Model name: encode all five seeds when an override is present so ablation runs
    # are distinguishable on disk; otherwise keep the name as-is (master convention).
    save_name = model_name(args.name, seeds)
    wandb_config.name = save_name
    train_config.model_save_path = "models/" + save_name

    print("\n")
    print("-"*20 + "ENSO RL AGENT TRAINING PIPELINE" + "-"*20)
    print(f"Seeds: {seeds.as_log_dict()}")
    print(f"Model name: {save_name}")
    
    try:
        start_time = time.time()

        # Setup
        obs_ds, train_ds, var_names, bounds, params = setup_environment(env_config)
        
        # Initialize W&B
        if wandb_config.mode != "disabled":
            initialize_wandb(wandb_config, train_config, env_config, seeds)
        
        # Create environment. seed_init (#4) and seed_physics (#5) pin the two
        # env-side randomness axes; reseed_on_reset=False so SB3's reset(seed=weight)
        # cannot couple them to the weight axis.
        print("\n4. Creating Gymnasium environment...")
        env = XROMultiYearEnv(
            params=params,
            train_ds=train_ds,
            var_names=var_names,
            max_steps=train_config.episode_length,
            seed_init=seeds.init,
            seed_physics=seeds.physics,
            reseed_on_reset=False,
        )
        print("\t[OK] Environment created")

        # Separate env for periodic mye_prob evaluation during training (kept distinct
        # so eval resets don't corrupt the PPO rollout). Left unseeded: it only feeds
        # the logged mye_prob curve and, thanks to the global-RNG isolation in
        # xro_step, cannot perturb the training shuffle or the trained model.
        eval_env = XROMultiYearEnv(
            params=params,
            train_ds=train_ds,
            var_names=var_names,
            max_steps=train_config.episode_length
        )

        # Train
        model = train_ppo_agent(env, train_config, wandb_config, seeds, eval_env=eval_env)
        
        # Evaluate
        evaluate_trained_model(env, model)
        
        # Finish W&B run
        if wandb_config.mode != "disabled":
            wandb.finish()
        
        elapsed = time.time() - start_time
        hours, remainder = divmod(int(elapsed), 3600)
        minutes, seconds = divmod(remainder, 60)
        print("\n" + "="*70)
        print(f" "*20 + f"TRAINING PIPELINE COMPLETED — Total time: {hours}h {minutes}m {seconds}s")
        print("="*70 + "\n")

    except Exception as e:
        elapsed = time.time() - start_time
        hours, remainder = divmod(int(elapsed), 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"\n[ERROR] Pipeline failed after {hours}h {minutes}m {seconds}s: {e}")
        import traceback
        traceback.print_exc()
        if wandb_config.mode != "disabled":
            wandb.finish(1)
        sys.exit(1)


if __name__ == "__main__":
    main()
