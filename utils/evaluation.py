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
    classified = classify_enso_event(enso_history, threshold=env.threshold, min_duration=24)
    mye_months = np.sum((classified == 'Multi-year El Nino') | (classified == 'Multi-year La Nina'))
    mye_probability = mye_months / len(classified)
    
    return mye_probability


def simulate_trajectory(env, agent=None, num_months=6000, disable_control_for_idx=None, 
                       debug_mode=False):
    """
    Simulate a continuous trajectory with optional action disabling.
    
    Args:
        env: Gymnasium environment
        agent: Trained agent (if None, uses zero actions)
        num_months (int): Duration of simulation
        disable_control_for_idx (int or None): Index of action to disable
        debug_mode (bool): Print debug info
        
    Returns:
        dict: Simulation data with trajectories and statistics
    """
    import time
    from utils.enso_classifier import classify_enso_event, summarize_classification
    
    start_time = time.perf_counter()
    
    # Get action_scale from config (single source of truth)
    env_config = EnvConfig()
    action_scale = np.array(env_config.action_scale)
    
    simulation_data = []
    obs, _ = env.reset()
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
        
        # Store scaled actions for consistent analysis
        scaled_action = action * action_scale
        actions_history.append(scaled_action)
        
        obs, reward, terminated, truncated, _ = env.step(action)
        total_rewards += reward
        sim_enso_history.append(obs[0])
        states_history.append(obs[:-1])
    
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    
    classified_event_array = classify_enso_event(sim_enso_history)
    classified_event = summarize_classification(classified_event_array)
    avg_reward = total_rewards / num_months
    
    # Calculate MYE probability as percentage of months in multi-year events
    mye_months = np.sum((classified_event_array == 'Multi-year El Nino') | 
                        (classified_event_array == 'Multi-year La Nina'))
    mye_prob = mye_months / len(classified_event_array)

    simulation_data = {
        'no_months': len(sim_enso_history) - 1,
        'enso_traj': np.array(sim_enso_history),
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
