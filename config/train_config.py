"""
Training configuration for PPO agent.
"""
from dataclasses import dataclass


@dataclass
class TrainConfig:
    """Training hyperparameters for PPO."""
    
    # Training duration (standard PPO knobs; each timestep is one month)
    total_timesteps: int = 240_000   # total env steps for model.learn() (= 1000 updates of n_steps)
    n_steps: int = 12 * 20           # rollout length collected per PPO update (240)
    n_epochs: int = 10               # PPO optimization passes over each rollout buffer (SB3 default)

    # Episode length / time limit (months). Resets every max_episode_steps for
    # start-state diversity; treated as truncation (partial-episode bootstrapping),
    # NOT termination. None = continuous (no resets).
    max_episode_steps: int = 1200    # 100 years

    # Learning rate
    learning_rate: float = 0.0003

    # Discount factor. ~100-month effective horizon at 0.99; 0.95 => ~20-month horizon
    gamma: float = 0.95

    # PPO value-function clipping range. Bounds how far the critic prediction can move
    # per update (PPO2-style), this and VecNormalize(norm_reward=True) in train.py tame value_loss 
    # blowup from the soft-constraint penalties. None disables it.
    clip_range_vf: float = 0.2

    # Random seed for PPO (reproducibility / multi-seed ensembles). None = unseeded.
    seed: int = None
    
    # Evaluation settings
    eval_steps: int = 240  # months per periodic in-training eval rollout

    # Debug mode
    debug_mode: bool = True

    # Model saving
    model_save_path: str = "models/rl_model"

    def __post_init__(self):
        """Validate configuration."""
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if self.total_timesteps < self.n_steps:
            raise ValueError("total_timesteps must be >= n_steps")
