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
    # Action scaling factors per variable per month (median absolute change from observational data)
    action_scale_median: list = field(default_factory=lambda: [
        # WWV
        [1.9820, 1.3861, 1.4365, 1.4435, 1.1293, 1.3021, 1.2335, 0.7784, 1.1000, 0.9740, 1.2712, 1.6017],
        # NPMM
        [0.1262, 0.1126, 0.0976, 0.1285, 0.1147, 0.1225, 0.1014, 0.0820, 0.0928, 0.0907, 0.1095, 0.1490],
        # SPMM
        [0.2168, 0.1650, 0.1794, 0.1764, 0.1657, 0.1171, 0.0934, 0.0821, 0.0902, 0.1185, 0.1311, 0.2426],
        # IOB
        [0.1031, 0.0813, 0.1105, 0.0754, 0.1039, 0.1208, 0.0870, 0.0663, 0.0661, 0.0689, 0.0809, 0.0859],
        # IOD
        [0.1864, 0.2267, 0.2025, 0.1894, 0.1998, 0.1725, 0.1931, 0.1801, 0.1756, 0.1566, 0.1679, 0.2084],
        # SIOD
        [0.1289, 0.1704, 0.1384, 0.1324, 0.1384, 0.1812, 0.1440, 0.1053, 0.1159, 0.1419, 0.1498, 0.1769],
        # TNA
        [0.1568, 0.1472, 0.1733, 0.1072, 0.1209, 0.1000, 0.1164, 0.0929, 0.0869, 0.0845, 0.1119, 0.0822],
        # ATL3
        [0.0961, 0.1054, 0.1682, 0.1323, 0.1564, 0.2769, 0.2219, 0.1911, 0.1555, 0.1575, 0.1267, 0.1622],
        # SASD
        [0.2482, 0.2678, 0.2373, 0.2033, 0.1483, 0.1462, 0.1647, 0.1318, 0.1185, 0.1810, 0.2137, 0.3544],
    ])

    # Observation space dimension
    obs_dim: int = 13  # 10 variables + month + ENSO-event duration + phase sign

    # Action space dimension
    action_dim: int = 9

    # Maximum steps for environment (None = continuous)
    max_steps: Optional[int] = None

    # Reward structure parameters (matches _calculate_reward in xro_env.py)
    reward_config: Dict = field(default_factory=lambda: {
        "action_penalty_weight": 0.002,
        "duration_reward_0_6m": 0.1,
        "duration_reward_7_12m": 0.3,
        "duration_reward_13m_plus": 1.0,  # full reward in the multi-year band
        # Phase-specific duration ceilings (months). Beyond the ceiling a soft
        # linear penalty ramps up to discourage *unrealistic persistence* — a
        # temporal failure mode the per-state Mahalanobis term cannot catch.
        # Ceilings = the observed ORAS5 maxima (El Nino 18mo, La Nina 25mo),
        # rounded up slightly so the real triple-dip La Nina (25mo) is still
        # fully rewarded but anything *longer than ever observed* is penalized.
        "duration_ceiling_el_nino": 24,
        "duration_ceiling_la_nina": 24,
        "duration_penalty_rate": 0.7,  # per month beyond the phase ceiling (sharp brake)
        # State-plausibility (Mahalanobis) penalty: catches *extreme/implausible
        # states* (e.g. Nino3.4 at -3 sigma, impossible mode combinations),
        # complementary to the duration brake above.
        "realism_penalty_weight": 0.05,
        "realism_quantile": 0.95,  # chi-squared quantile (dof = n_modes) for activation
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

        self.action_scale = [[v / 2 for v in row] for row in self.action_scale]