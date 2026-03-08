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
    
    # Action scaling factors per variable per month (mean absolute change from observational data)
    # Shape: 9 variables × 12 months [Jan=0 ... Dec=11]
    # Variables: WWV, NPMM, SPMM, IOB, IOD, SIOD, TNA, ATL3, SASD
    action_scale: list = field(default_factory=lambda: [
        # WWV - Jan through Dec
        [2.4086, 1.6970, 1.7290, 1.5935, 1.3136, 1.5871, 1.3944, 1.0521, 1.3286, 1.0427, 1.3655, 1.8750],
        # NPMM
        [0.1640, 0.1357, 0.1201, 0.1402, 0.1565, 0.1295, 0.1150, 0.0866, 0.1025, 0.1061, 0.1399, 0.1540],
        # SPMM
        [0.2509, 0.2091, 0.2008, 0.1839, 0.1861, 0.1393, 0.1162, 0.1122, 0.1210, 0.1381, 0.1545, 0.2498],
        # IOB
        [0.1106, 0.1189, 0.1179, 0.0962, 0.1191, 0.1293, 0.0929, 0.0808, 0.0772, 0.0838, 0.0920, 0.1023],
        # IOD
        [0.2284, 0.2663, 0.2310, 0.2218, 0.2602, 0.2252, 0.2179, 0.2369, 0.2405, 0.2223, 0.2533, 0.2776],
        # SIOD
        [0.1569, 0.2402, 0.1997, 0.1803, 0.1737, 0.1864, 0.1595, 0.1140, 0.1335, 0.1601, 0.2099, 0.2226],
        # TNA
        [0.1778, 0.1685, 0.2015, 0.1312, 0.1322, 0.1253, 0.1436, 0.1205, 0.1007, 0.0958, 0.1269, 0.0998],
        # ATL3
        [0.1296, 0.1499, 0.1761, 0.1753, 0.2160, 0.3046, 0.2731, 0.2272, 0.2094, 0.1905, 0.1674, 0.1761],
        # SASD
        [0.3412, 0.3235, 0.3208, 0.2567, 0.1933, 0.1749, 0.1943, 0.1585, 0.1343, 0.2089, 0.2865, 0.3701],
    ])
    
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
            raise ValueError(f"action_scale must have {self.action_dim} variables")
        if any(len(row) != 12 for row in self.action_scale):
            raise ValueError("Each action_scale row must have 12 monthly values")
        if self.action_dim <= 0 or self.obs_dim <= 0:
            raise ValueError("action_dim and obs_dim must be positive")
