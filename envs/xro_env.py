"""
XRO ENSO Environment for Gymnasium.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from utils.physics import xro_step
from utils.data_processing import get_data


class XROMultiYearEnv(gym.Env):
    """
    CONTINUOUS CLIMATE ENVIRONMENT for ENSO control.
    - No artificial episode boundaries
    - Agent learns long-term climate control
    - Preserves multi-year ENSO memory and continuity
    """
    
    def __init__(self, params, train_ds, var_names, max_steps=None):
        """
        Initialize XRO environment.
        
        Args:
            params (dict): Parameters from data processing (model, fit_ds, etc.)
            train_ds: Training dataset (xarray)
            var_names (list): Variable names
            max_steps (int or None): Maximum steps per episode (None = continuous)
        """
        super(XROMultiYearEnv, self).__init__()
        self.params = params
        self.train_ds = train_ds
        self.var_names = var_names
        self.n_modes = len(var_names)
        self.max_steps = max_steps
        self.current_step = 0
        self.rng = np.random.default_rng()
        
        # Action space: 9D continuous [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(9,),
            dtype=np.float32
        )
        
        # Observation space: State (10D) + Month feature (1D) = 11D
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_modes + 1,),
            dtype=np.float32
        )
        
        self.enso_history = []
        self.consecutive_enso_months = 0
        self.threshold = params.get('threshold', 1.0)

    def reset(self, seed=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)

        # Generate random starting date
        random_year = self.rng.integers(1979, 2023)
        random_month = self.rng.integers(1, 13)
        self.state = np.array(get_data(self.train_ds, self.var_names, random_year, random_month), dtype=np.float32)

        self.current_step = 0
        self.enso_history = [self.state[0]]
        self.consecutive_enso_months = 0
        
        return self._get_obs(), {}

    def _get_obs(self):
        """Get observation (state + month feature)."""
        month_feature = (self.current_step % 12) / 12.0
        return np.concatenate([self.state, [month_feature]], dtype=np.float32)

    def step(self, action):
        """
        Execute one step in the environment.
        
        Args:
            action (np.ndarray): 9D action vector
            
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Step the model
        self.state = xro_step(self.state, self.params, action, self.rng, self.current_step)
        self.enso_history.append(self.state[0])
        self.current_step += 1
        
        # Check termination
        terminated = False
        if self.max_steps is not None and self.current_step >= self.max_steps:
            terminated = True
        
        # Calculate reward
        reward = self._calculate_reward(action)
        
        return self._get_obs(), reward, terminated, False, {}

    def _calculate_reward(self, action):
        """
        Calculate reward based on ENSO state and duration.
        
        Args:
            action (np.ndarray): Control actions taken
            
        Returns:
            float: Reward value
        """
        enso_index = self.state[0]
        threshold = self.threshold
        
        # Initialize reward components
        enso_reward = 0.0
        action_penalty = -0.005 * np.sum(np.abs(action) ** 2)
        duration_reward = 0.0
        duration_penalty = 0.0
        
        # ENSO reward and duration tracking
        is_enso_month = False
        if enso_index > threshold:
            enso_reward = 0.1
            self.consecutive_enso_months += 1
            is_enso_month = True
        elif enso_index < -threshold:
            enso_reward = 0.1
            self.consecutive_enso_months += 1
            is_enso_month = True
        else:
            self.consecutive_enso_months = 0
        
        # Duration reward (scales with event length)
        if is_enso_month and self.consecutive_enso_months <= 36:
            if self.consecutive_enso_months >= 24:
                duration_reward = 1.0
            elif self.consecutive_enso_months >= 18:
                duration_reward = 0.6
            elif self.consecutive_enso_months >= 12:
                duration_reward = 0.4
            elif self.consecutive_enso_months >= 6:
                duration_reward = 0.2
        
        # Duration penalty for unrealistic persistence
        if self.consecutive_enso_months > 36:
            duration_penalty = -0.1 * (self.consecutive_enso_months - 36)
        
        # Combine all components
        total_reward = enso_reward + duration_reward + duration_penalty #+ action_penalty
        
        return float(total_reward)

    def _check_multi_year_event(self, history):
        """
        Check if history contains at least 2 consecutive years of ENSO activity.
        
        Args:
            history (list or np.ndarray): ENSO history
            
        Returns:
            bool: True if multi-year event detected
        """
        history = np.array(history)
        threshold = self.threshold
        min_duration = 24
        
        # El Nino check
        is_nino = (history >= threshold).astype(int)
        if self._max_run_length(is_nino) >= min_duration:
            return True
        
        # La Nina check
        is_nina = (history <= -threshold).astype(int)
        if self._max_run_length(is_nina) >= min_duration:
            return True
        
        return False

    @staticmethod
    def _max_run_length(binary_sequence):
        """Calculate longest consecutive run of 1s in binary sequence."""
        padded = np.concatenate(([0], binary_sequence, [0]))
        changes = np.diff(padded)
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]
        
        if len(starts) == 0:
            return 0
        
        return (ends - starts).max()
