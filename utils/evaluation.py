"""
Evaluation utilities for ENSO RL agent.
"""
import numpy as np
from config.env_config import EnvConfig


def evaluate_agent(env, agent=None, continuous_steps=6000):
    """
    Evaluate agent performance over continuous steps.
    
    Args:
        env: Gymnasium environment
        agent: Trained agent (if None, uses zero actions)
        continuous_steps (int): Number of steps to evaluate
        
    Returns:
        float: Probability of multi-year events (percentage of months in 24+ month events)
    """
    from utils.enso_classifier import classify_enso_event
    
    enso_history = []
    obs, _ = env.reset()
    enso_history.append(obs[0])
    
    for step in range(continuous_steps):
        if agent:
            action, _ = agent.predict(obs, deterministic=True)
        else:
            action = np.zeros(9)
        
        obs, reward, terminated, truncated, _ = env.step(action)
        enso_history.append(obs[0])
    
    # Classify events and calculate multi-year probability
    classified = classify_enso_event(enso_history, threshold=env.threshold, min_duration=12)
    mye_months = np.sum((classified == 'Multi-year El Nino') | (classified == 'Multi-year La Nina'))
    mye_probability = mye_months / len(classified)
    
    return mye_probability


def rollout_mye_phased(env, agent=None, num_months=1200, seed=None):
    """Single rollout returning multi-year-ENSO probability split by phase.

    Used by the lift analysis (paper points 1.1 quantified lift and 1.2 El Nino
    vs La Nina separation). Pass the SAME seed for the agent and baseline rollouts
    to get a paired comparison (identical start state and noise sequence).

    Args:
        env: Gymnasium environment
        agent: Trained agent (if None, uses zero actions = free-running baseline)
        num_months (int): Rollout length in months
        seed (int or None): Reset seed. Fixes initial conditions AND noise.

    Returns:
        dict: {'total', 'el_nino', 'la_nina'} — fraction of months in a
              multi-year event of each type.
    """
    from utils.enso_classifier import classify_enso_event, mye_fraction_by_phase

    obs, _ = env.reset(seed=seed)
    enso_history = [obs[0]]
    for _ in range(num_months):
        if agent is not None:
            action, _ = agent.predict(obs, deterministic=True)
        else:
            action = np.zeros(env.action_space.shape)
        obs, _, _, _, _ = env.step(action)
        enso_history.append(obs[0])

    classified = classify_enso_event(enso_history, threshold=env.threshold)
    return mye_fraction_by_phase(classified)


def rollout_mye_events(env, agent=None, num_months=1200, seed=None, min_duration=12):
    """Single rollout returning, per phase, the multi-year-ENSO time-fraction PLUS
    the count and per-event durations (months) of multi-year events.

    This disentangles the two ingredients of mye_prob (time-fraction ≈ event
    frequency × mean duration): does the agent raise P(MYE) by making events
    MORE FREQUENT or LONGER? A multi-year event is a continuous same-sign run of
    |Nino3.4|>threshold lasting > min_duration months (matches classify_enso_event).

    Returns dict phase -> {'frac', 'count', 'durations'(list of months)}.
    """
    from utils.enso_classifier import (classify_enso_event, mye_fraction_by_phase,
                                        _find_continuous_runs)

    obs, _ = env.reset(seed=seed)
    enso_history = [obs[0]]
    for _ in range(num_months):
        if agent is not None:
            action, _ = agent.predict(obs, deterministic=True)
        else:
            action = np.zeros(env.action_space.shape)
        obs, _, _, _, _ = env.step(action)
        enso_history.append(obs[0])

    enso = np.asarray(enso_history)
    thr = env.threshold
    frac = mye_fraction_by_phase(classify_enso_event(enso, threshold=thr))

    out = {}
    binmap = {'el_nino': (enso >= thr).astype(int), 'la_nina': (enso <= -thr).astype(int)}
    for ph, b in binmap.items():
        durs = [length for _s, _e, length in _find_continuous_runs(b) if length > min_duration]
        out[ph] = {'frac': frac[ph], 'count': len(durs), 'durations': durs}
    out['total'] = {'frac': frac['total'],
                    'count': out['el_nino']['count'] + out['la_nina']['count'],
                    'durations': out['el_nino']['durations'] + out['la_nina']['durations']}
    return out


def simulate_trajectory(env, agent=None, num_months=6000, disable_control_for_idx=None,
                       debug_mode=False, seed=None):
    """
    Simulate a continuous trajectory with optional action disabling.
    
    Args:
        env: Gymnasium environment
        agent: Trained agent (if None, uses zero actions)
        num_months (int): Duration of simulation
        disable_control_for_idx (int or None): Index of action to disable
        debug_mode (bool): Print debug info
        seed (int or None): Random seed for env.reset(). Fixes initial conditions
              AND noise sequence. Use the same seed across baseline + interventions
              for paired comparison.
        
    Returns:
        dict: Simulation data with trajectories and statistics
    """
    import time
    from utils.enso_classifier import classify_enso_event, summarize_classification
    
    start_time = time.perf_counter()
    
    # Get action_scale from config (single source of truth)
    env_config = EnvConfig()
    action_scale_matrix = np.array(env_config.action_scale)  # shape: (9, 12)
    
    simulation_data = []
    obs, _ = env.reset(seed=seed)
    sim_enso_history = [obs[0]]
    actions_history = []
    states_history = [obs[:-1]]
    total_rewards = 0.0
    
    if debug_mode:
        if disable_control_for_idx is not None:
            print(f"Starting Simulation with disabled action {disable_control_for_idx} for {num_months/12:.1f} years...")
        else:
            print(f"Starting Simulation for {num_months/12:.1f} years...")

    for step in range(num_months):
        if agent:
            action, _ = agent.predict(obs, deterministic=True)
        else:
            action = np.zeros(env.action_space.shape)
        
        if disable_control_for_idx is not None:
            action[disable_control_for_idx] = 0.0
        
        # Store scaled actions for consistent analysis (use current month's scale)
        current_month = step % 12
        action_scale = action_scale_matrix[:, current_month]
        scaled_action = action * action_scale
        actions_history.append(scaled_action)
        
        obs, reward, terminated, truncated, _ = env.step(action)
        total_rewards += reward
        sim_enso_history.append(obs[0])
        states_history.append(obs[:-1])
    
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    
    classified_event_array = classify_enso_event(sim_enso_history, threshold=env.threshold)
    classified_event = summarize_classification(classified_event_array)
    avg_reward = total_rewards / num_months
    
    # Calculate MYE probability as percentage of months in multi-year events
    mye_months = np.sum((classified_event_array == 'Multi-year El Nino') | 
                        (classified_event_array == 'Multi-year La Nina'))
    mye_prob = mye_months / len(classified_event_array)

    simulation_data = {
        'no_months': len(sim_enso_history) - 1,
        'enso_traj': np.array(sim_enso_history), # 3 month mean of xro simulates nino
        'actions_traj': np.array(actions_history),
        'states_traj': np.array(states_history),
        'classified_event': classified_event,  # Summary string
        'classified_event_array': classified_event_array,  # Detailed month-by-month
        'avg_reward': avg_reward,
        'mye_probability': mye_prob,  # Percentage of months in multi-year events
        'mye_months': mye_months,  # Actual count of multi-year months
        'elapsed_time': elapsed,
    }
    
    if debug_mode:
        print(f"[OK] Simulated {len(sim_enso_history)-1} months in {elapsed:.2f}s")
        print(f"Avg Reward: {avg_reward:.4f}")

    return simulation_data
