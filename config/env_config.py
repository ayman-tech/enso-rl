"""
Environment configuration for XRO ENSO system.
"""
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional


@dataclass
class EnvConfig:
    """Configuration for XROMultiYearEnv."""
    
    # ENSO threshold for determining events
    threshold: float = 0.5
    
    # Action scaling factors for each controllable variable
    # (WWV, NPMM, SPMM, IOB, IOD, SIOD, TNA, ATL3, SASD)
    action_scale: list = field(default_factory=lambda: [1.8, 0.4, 0.3, 0.3, 0.7, 0.4, 0.35, 0.6, 0.4])
    
    # Observation space dimension
    obs_dim: int = 11  # 10 variables + 1 month feature
    
    # Action space dimension
    action_dim: int = 9
    
    # Maximum steps for environment (None = continuous)
    max_steps: Optional[int] = None
    
    # Reward structure parameters
    reward_config: Dict = field(default_factory=lambda: {
        "enso_reward": 0.1,
        "action_penalty_weight": 0.01,
        "multi_year_event_reward": 0.5,
        "min_duration_multi_year": 24,  # months
        "duration_reward_threshold_6m": 0.10,
        "duration_reward_threshold_12m": 0.15,
        "duration_reward_threshold_18m": 0.25,
        "duration_reward_threshold_24m": 0.30,
        "duration_penalty_start": 30,  # months
        "duration_penalty_rate": 0.01,
        "duration_penalty_steeper_start": 36,  # months
        "duration_penalty_steeper_rate": 0.02,
    })
    
    # Data configuration
    data_config: Dict = field(default_factory=lambda: {
        "data_path": "data/XRO_indices_oras5.nc",
        "train_start": "1979-01",
        "train_end": "2022-12",
    })
    
    # XRO model configuration
    # xro_config: Dict = field(default_factory=lambda: {
    #     "maskb": ,
    #     "maskNT": ,
    # })
    
    def __post_init__(self):
        """Validate configuration."""
        if self.threshold <= 0:
            raise ValueError("Threshold must be positive")
        if len(self.action_scale) != self.action_dim:
            raise ValueError(f"action_scale must have {self.action_dim} elements")
        if self.action_dim <= 0 or self.obs_dim <= 0:
            raise ValueError("action_dim and obs_dim must be positive")
        
        # self.action_scale = [round(x/12,4) for x in self.action_scale]
