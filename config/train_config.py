"""
Training configuration for PPO agent.
"""
from dataclasses import dataclass


@dataclass
class TrainConfig:
    """Training hyperparameters for PPO."""
    
    # Training duration
    n_steps: int = 12 * 20 # steps before each update
    train_epochs: int = 1000
    train_months: int = n_steps*train_epochs  # episodes to train for
    
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
    
    # Simulation settings
    sim_months: int = 240  # simulation months for evaluation
    episode_length: int = None  # Reset after episode_length. Set to None for continuous (no reset)
    
    # Debug mode
    debug_mode: bool = True
    
    # Model saving
    model_save_path: str = "models/rl_model"
    
    def __post_init__(self):
        """Validate configuration."""
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if self.train_months < self.n_steps:
            raise ValueError("train_epochs should be >= n_steps")
