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
    
    def __init__(self, params, train_ds, var_names, max_steps=None,
                 seed_init=None, seed_physics=None, reseed_on_reset=True):
        """
        Initialize XRO environment.

        Args:
            params (dict): Parameters from data processing (model, fit_ds, etc.)
            train_ds: Training dataset (xarray)
            var_names (list): Variable names
            max_steps (int or None): Maximum steps per episode (None = continuous)
            seed_init (int or None): Seed for the start-state sampling stream
                (randomness axis #4). Independent of seed_physics so the two axes can
                be studied separately.
            seed_physics (int or None): Seed for the XRO climate-noise stream
                (randomness axis #5). Each step draws one integer from this generator
                to seed XRO.simulate.
            reseed_on_reset (bool): If True (default), reset(seed=X) re-derives both
                generators deterministically from X, so reset(seed) fixes both the
                start state AND the noise sequence — the contract the paired-comparison
                analyses in utils/evaluation.py depend on. If False (training env),
                reset ignores the passed seed and the constructor-seeded generators
                persist, so SB3's reset(seed=weight_seed) cannot couple axes #4/#5 to
                the weight axis.
        """
        super(XROMultiYearEnv, self).__init__()
        self.params = params
        self.train_ds = train_ds
        self.var_names = var_names
        self.n_modes = len(var_names)
        self.max_steps = max_steps
        self.current_step = 0
        self.month_offset = 0  # set per-reset to the sampled state's real month
        # Independent RNG streams for the two environment randomness axes.
        self.rng_init = np.random.default_rng(seed_init)        # axis #4: start state
        self.rng_physics = np.random.default_rng(seed_physics)  # axis #5: climate noise
        self._reseed_on_reset = reseed_on_reset
        
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
        self.enso_phase_sign = 0  # +1 El Nino, -1 La Nina, 0 neutral
        self.threshold = params.get('threshold', 0.5)

        # Reward weights (defaults mirror EnvConfig.reward_config)
        reward_config = params.get('reward_config', {})
        self.action_penalty_weight = reward_config.get('action_penalty_weight', 0.002)
        self.realism_penalty_weight = reward_config.get('realism_penalty_weight', 0.05)
        realism_quantile = reward_config.get('realism_quantile', 0.95)
        # Phase-specific duration ceilings + ramp (discourage unrealistic persistence)
        self.duration_ceiling_el_nino = reward_config.get('duration_ceiling_el_nino', 24)
        self.duration_ceiling_la_nina = reward_config.get('duration_ceiling_la_nina', 36)
        self.duration_penalty_rate = reward_config.get('duration_penalty_rate', 0.3)

        # State-plausibility (Mahalanobis) reference from observed climatology.
        # Catches extreme/implausible single states; the duration ramp above
        # handles the temporal (over-persistence) failure mode it cannot see.
        self._init_realism_reference(train_ds, var_names, realism_quantile)

    def _init_realism_reference(self, train_ds, var_names, quantile):
        """Precompute mean, inverse covariance, and chi-squared activation
        threshold of the observed state distribution for the Mahalanobis penalty."""
        from scipy.stats import chi2

        # Stack observed states into [n_time, n_modes] and drop incomplete rows
        obs_states = np.column_stack([
            np.asarray(train_ds[name].values, dtype=np.float64) for name in var_names
        ])
        obs_states = obs_states[~np.isnan(obs_states).any(axis=1)]

        self._realism_mean = obs_states.mean(axis=0)
        cov = np.cov(obs_states, rowvar=False)
        # Small ridge for numerical stability of the inverse
        cov += np.eye(cov.shape[0]) * 1e-6
        self._realism_inv_cov = np.linalg.inv(cov)
        # Activation threshold: states beyond this chi-squared quantile are penalized
        self._realism_threshold = float(chi2.ppf(quantile, df=len(var_names)))

    def reset(self, seed=None):
        """Reset environment to initial state.

        When reseed_on_reset is True and an explicit seed is given, both the
        start-state and physics generators are re-derived deterministically from it
        (two independent sub-streams), so the same seed fixes both the start state and
        the noise sequence — required for the paired baseline/intervention analyses.
        When False (training env), the constructor-seeded generators persist and the
        passed seed is ignored, keeping axes #4/#5 decoupled from SB3's reset seed.
        """
        super().reset(seed=seed)
        if self._reseed_on_reset and seed is not None:
            ci, cp = np.random.SeedSequence(seed).spawn(2)
            self.rng_init = np.random.default_rng(ci)
            self.rng_physics = np.random.default_rng(cp)

        # Generate random starting date
        random_year = self.rng_init.integers(1979, 2023)
        random_month = self.rng_init.integers(1, 13)
        self.state = np.array(get_data(self.train_ds, self.var_names, random_year, random_month), dtype=np.float32)

        # Align the seasonal clock to the sampled state's real calendar month so
        # the state evolves under its true season (e.g. an August state advances
        # through Sep, Oct, ...) instead of being pinned to January. ENSO is
        # strongly seasonally phase-locked, so injecting a real state at the
        # wrong calendar phase is physically inconsistent. month_offset shifts
        # current_step (0-based) onto the real month (also 0-based).
        self.month_offset = int(random_month) - 1

        self.current_step = 0
        self.enso_history = [self.state[0]]
        self.consecutive_enso_months = 0
        self.enso_phase_sign = 0

        return self._get_obs(), {}

    def _get_obs(self):
        """Get observation (state + month feature)."""
        month_feature = ((self.current_step + self.month_offset) % 12) / 12.0
        return np.concatenate([self.state, [month_feature]], dtype=np.float32)

    def step(self, action):
        """
        Execute one step in the environment.
        
        Args:
            action (np.ndarray): 9D action vector
            
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Step the model. Pass the season-aligned calendar index so the XRO
        # seasonal parameters match the state's real month (see month_offset).
        self.state = xro_step(self.state, self.params, action, self.rng_physics,
                              self.current_step + self.month_offset)
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
        action_penalty = -self.action_penalty_weight * np.sum(action ** 2)
        duration_reward = 0.0

        # ENSO duration tracking (sign-aware: a phase flip starts a new event,
        # so an El Nino -> La Nina swing is not counted as one continuous event)
        if enso_index > threshold:
            current_sign = 1
        elif enso_index < -threshold:
            current_sign = -1
        else:
            current_sign = 0

        if current_sign == 0:
            # Neutral: event ends
            self.consecutive_enso_months = 0
        elif current_sign == self.enso_phase_sign:
            # Same phase persists
            self.consecutive_enso_months += 1
        else:
            # New event of the opposite (or first) phase
            self.consecutive_enso_months = 1
        self.enso_phase_sign = current_sign
        
        # Duration reward: full in the multi-year band, then a soft per-phase
        # over-persistence penalty beyond the observed ceiling. The ramp (not a
        # cliff) keeps realistic long events fully rewarded while making
        # physically impossible persistence (e.g. a 100-year La Nina) unviable —
        # a temporal failure the per-state Mahalanobis term cannot detect.
        duration_penalty = 0.0
        if self.consecutive_enso_months > 0:
            if self.consecutive_enso_months <= 6:
                duration_reward = 0.1    # very soft: event just starting
            elif self.consecutive_enso_months <= 12:
                duration_reward = 0.3     # soft: approaching multi-year
            else:
                duration_reward = 1.0     # multi-year event (>=13 months)

            # Phase-specific over-persistence penalty
            ceiling = (self.duration_ceiling_el_nino if current_sign > 0
                       else self.duration_ceiling_la_nina)
            if self.consecutive_enso_months > ceiling:
                over = self.consecutive_enso_months - ceiling
                duration_penalty = -self.duration_penalty_rate * over

        # State-plausibility (Mahalanobis) penalty: discourage extreme/implausible
        # single states regardless of duration. Zero inside the observed envelope.
        realism_penalty = self._realism_penalty()

        # Combine all components
        total_reward = duration_reward + duration_penalty + realism_penalty + action_penalty

        return float(total_reward)

    def _realism_penalty(self):
        """Penalize states outside the observed multivariate envelope.

        Returns 0 when the squared Mahalanobis distance of the current state
        from the observed climatology is within the chi-squared threshold, and
        a negative penalty proportional to the excess otherwise.
        """
        delta = self.state - self._realism_mean
        d2 = float(delta @ self._realism_inv_cov @ delta)
        excess = max(0.0, d2 - self._realism_threshold)
        return -self.realism_penalty_weight * excess

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
