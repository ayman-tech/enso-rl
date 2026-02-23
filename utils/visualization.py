"""
Visualization utilities for ENSO RL agent training and evaluation.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wandb


def create_plots_directory():
    """Create plots directory if it doesn't exist."""
    os.makedirs("plots", exist_ok=True)


def save_and_log_plot(fig, plot_name, wandb_enabled=False):
    """
    Save figure to disk and optionally log to W&B.
    
    Args:
        fig: Matplotlib figure object
        plot_name (str): Name of the plot (without extension)
        wandb_enabled (bool): Whether to log to W&B
        
    Returns:
        str: Path to saved plot
    """
    create_plots_directory()
    file_path = f"plots/{plot_name}.png"
    fig.savefig(file_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # Log to W&B if enabled
    if wandb_enabled and wandb.run is not None:
        wandb.log({f"plots/{plot_name}": wandb.Image(file_path)})
    
    return file_path


def plot_enso_trajectory(enso_traj, threshold=1.0, num_months=None, wandb_enabled=False):
    """
    Plot ENSO index trajectory with thresholds.
    
    Args:
        enso_traj (array): ENSO index time series
        threshold (float): Threshold value for El Niño/La Niña
        num_months (int): Total number of months (for label)
        wandb_enabled (bool): Log to W&B
        
    Returns:
        str: Path to saved plot
    """
    if num_months is None:
        num_months = len(enso_traj)
    
    fig, ax = plt.subplots(figsize=(20, 6))
    
    ax.plot(enso_traj, linewidth=1.5, label='ENSO Index (Continuous)', color='black')
    ax.axhline(threshold, color='r', linestyle='--', label='El Niño Threshold', alpha=0.7, linewidth=2)
    ax.axhline(-threshold, color='b', linestyle='--', label='La Niña Threshold', alpha=0.7, linewidth=2)
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    
    ax.set_title("ENSO Index Trajectory (Continuous Control)", fontsize=14, fontweight='bold')
    ax.set_xlabel(f"Time (months) - Total {num_months} months", fontsize=12)
    ax.set_ylabel("Nino34 Index", fontsize=12)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    return save_and_log_plot(fig, "01_enso_trajectory", wandb_enabled)


def plot_control_actions(actions_traj, var_names, num_months=None, wandb_enabled=False):
    """
    Plot control actions for each variable.
    
    Args:
        actions_traj (array): Actions trajectory (num_steps x num_actions)
        var_names (list): Variable names (including Nino34)
        num_months (int): Total number of months
        wandb_enabled (bool): Log to W&B
        
    Returns:
        str: Path to saved plot
    """
    if num_months is None:
        num_months = actions_traj.shape[0]
    
    action_vars = var_names[1:]  # Skip Nino34
    num_actions = len(action_vars)
    
    fig, axes = plt.subplots(nrows=num_actions, ncols=1, figsize=(20, num_actions * 2.5), sharex=True)
    if num_actions == 1:
        axes = [axes]
    
    fig.suptitle("Continuous Control Actions for Each Variable", fontsize=14, fontweight='bold')
    
    for i, var_name in enumerate(action_vars):
        ax = axes[i]
        ax.plot(actions_traj[:, i], linewidth=0.8, color='steelblue')
        ax.fill_between(range(len(actions_traj)), actions_traj[:, i], alpha=0.2, color='steelblue')
        ax.set_title(f'{var_name} (Mean: {np.mean(actions_traj[:, i]):.4f}, Std: {np.std(actions_traj[:, i]):.4f})', 
                     fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
        
        if i == 0:
            ax.set_ylabel("Action Value", fontsize=10)
    
    axes[-1].set_xlabel(f"Time (months) - Total {num_months} months", fontsize=11)
    fig.tight_layout()
    
    return save_and_log_plot(fig, "02_control_actions", wandb_enabled)


def plot_state_variables(states_traj, var_names, threshold=1.0, num_months=None, wandb_enabled=False):
    """
    Plot state variables trajectory.
    
    Args:
        states_traj (array): States trajectory (num_steps x num_variables)
        var_names (list): Variable names
        threshold (float): ENSO threshold
        num_months (int): Total number of months
        wandb_enabled (bool): Log to W&B
        
    Returns:
        str: Path to saved plot
    """
    if num_months is None:
        num_months = states_traj.shape[0]
    
    num_vars = len(var_names)
    fig, axes = plt.subplots(nrows=num_vars, ncols=1, figsize=(20, num_vars * 2.5), sharex=True)
    if num_vars == 1:
        axes = [axes]
    
    fig.suptitle("State Variables Trajectory (Continuous Control)", fontsize=14, fontweight='bold')
    
    for i, var_name in enumerate(var_names):
        ax = axes[i]
        ax.plot(states_traj[:, i], linewidth=0.8, color='green', alpha=0.8)
        ax.fill_between(range(len(states_traj)), states_traj[:, i], alpha=0.2, color='green')
        
        # Add thresholds for ENSO index
        if var_name == 'Nino34':
            ax.axhline(threshold, color='r', linestyle='--', alpha=0.6, label='El Niño Thr')
            ax.axhline(-threshold, color='b', linestyle='--', alpha=0.6, label='La Niña Thr')
            ax.legend(loc='upper right', fontsize=9)
        
        ax.set_title(f'{var_name} (Mean: {np.mean(states_traj[:, i]):.4f}, Std: {np.std(states_traj[:, i]):.4f})', 
                     fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
        
        if i == 0:
            ax.set_ylabel("Variable Value", fontsize=10)
    
    axes[-1].set_xlabel(f"Time (months) - Total {num_months} months", fontsize=11)
    fig.tight_layout()
    
    return save_and_log_plot(fig, "03_state_variables", wandb_enabled)


def plot_feature_importance(delta_r_values, wandb_enabled=False):
    """
    Plot feature importance from interventional analysis.
    
    Args:
        delta_r_values (list): List of dicts with 'feature' and 'delta_r' keys
        wandb_enabled (bool): Log to W&B
        
    Returns:
        str: Path to saved plot
    """
    features = [item['feature'] for item in delta_r_values]
    delta_r_vals = [item['delta_r'] for item in delta_r_values]
    
    # Sort by delta_r
    sorted_idx = np.argsort(delta_r_vals)
    features = [features[i] for i in sorted_idx]
    delta_r_vals = [delta_r_vals[i] for i in sorted_idx]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Color scheme: red for negative (strong drivers), blue for positive
    colors = ['red' if dr < 0 else 'blue' for dr in delta_r_vals]
    bars = ax.bar(features, delta_r_vals, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
    
    ax.axhline(0, color='gray', linestyle='--', linewidth=2)
    
    ax.set_xlabel('Controllable Features', fontsize=12, fontweight='bold')
    ax.set_ylabel('Delta R (Impact on Reward)', fontsize=12, fontweight='bold')
    ax.set_title('Interventional Driver Analysis: Impact of Disabling Control Actions', 
                 fontsize=13, fontweight='bold')
    
    # Legend
    red_patch = plt.Line2D([0], [0], marker='s', color='w', label='Negative ΔR (Strong Driver)',
                          markerfacecolor='red', markersize=10)
    blue_patch = plt.Line2D([0], [0], marker='s', color='w', label='Positive ΔR (Weak Driver)',
                           markerfacecolor='blue', markersize=10)
    ax.legend(handles=[red_patch, blue_patch], fontsize=10, loc='upper left')
    
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.xticks(rotation=45, ha='right')
    fig.tight_layout()
    
    return save_and_log_plot(fig, "04_feature_importance", wandb_enabled)


def plot_event_classification(enso_traj, classifications_array, num_months=None, wandb_enabled=False):
    """
    Plot ENSO trajectory with month-by-month event classification.
    
    Args:
        enso_traj (array): ENSO index time series
        classifications_array (array): Month-by-month event classifications
        num_months (int): Total number of months
        wandb_enabled (bool): Log to W&B
        
    Returns:
        str: Path to saved plot
    """
    if num_months is None:
        num_months = len(enso_traj)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 8), height_ratios=[3, 1])
    
    # Plot 1: ENSO index with thresholds
    ax1.plot(enso_traj, linewidth=1.5, label='ENSO Index', color='black', zorder=3)
    ax1.set_xlim(0, len(enso_traj))
    
    ax1.axhline(1.0, color='r', linestyle='--', alpha=0.5, linewidth=1.5, label='El Niño Threshold')
    ax1.axhline(-1.0, color='b', linestyle='--', alpha=0.5, linewidth=1.5, label='La Niña Threshold')
    ax1.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax1.set_title("ENSO Index with Month-by-Month Event Classification", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Nino34 Index", fontsize=12)
    ax1.legend(loc='upper right', fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Classification timeline
    color_map = {
        'Neutral': '#95B8D1',                    # Light blue
        'Single-year El Nino': '#F4A460',        # Sandy brown
        'Single-year La Nina': '#4682B4',        # Steel blue
        'Multi-year El Nino': '#FF6347',         # Tomato red
        'Multi-year La Nina': '#1E90FF',         # Dodger blue
    }
    
    # Create color array
    colors = np.array([color_map.get(class_name, '#CCCCCC') for class_name in classifications_array])
    
    # Plot as bars
    ax2.bar(range(len(classifications_array)), np.ones(len(classifications_array)), 
            color=colors, width=1, edgecolor='none', alpha=0.9)
    
    ax2.set_xlim(-0.5, len(classifications_array) - 0.5)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel(f"Time (months) - Total {num_months} months", fontsize=12)
    ax2.set_yticks([])
    ax2.set_ylabel("Event Type", fontsize=11)
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add legend on the side
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color_map[key], label=key) 
                      for key in color_map.keys()]
    ax2.legend(handles=legend_elements, loc='upper center', ncol=3, 
              bbox_to_anchor=(0.5, -0.3), frameon=True, fontsize=9)
    
    fig.tight_layout()
    return save_and_log_plot(fig, "07_event_classification", wandb_enabled)


def create_evaluation_summary_plots(evaluation_results, wandb_enabled=False):
    """
    Create summary comparison plots.
    
    Args:
        evaluation_results (dict): Results from evaluation with keys:
            - evaluation/multi_year_events_with_agent
            - evaluation/multi_year_events_without_agent
            - evaluation/improvement_percentage_points
            - evaluation/improvement_ratio
        wandb_enabled (bool): Log to W&B
        
    Returns:
        str: Path to saved plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Agent vs Baseline Performance", fontsize=14, fontweight='bold')
    
    # Performance comparison
    ax = axes[0]
    methods = ['With Agent', 'Without Agent (Baseline)']
    values = [
        evaluation_results['evaluation/multi_year_events_with_agent'],
        evaluation_results['evaluation/multi_year_events_without_agent']
    ]
    colors_bars = ['green', 'red']
    bars = ax.bar(methods, values, color=colors_bars, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel('Multi-Year Event Probability', fontsize=11, fontweight='bold')
    ax.set_title('Event Probability Comparison', fontsize=12, fontweight='bold')
    ax.set_ylim(0, max(values) * 1.2)
    
    # Add value labels
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{value:.2%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.grid(axis='y', alpha=0.3)
    
    # Improvement metrics
    ax = axes[1]
    improvement_pp = evaluation_results['evaluation/improvement_percentage_points']
    improvement_ratio = evaluation_results['evaluation/improvement_ratio']
    
    metrics = ['Improvement\n(pp)', 'Improvement\nRatio (x)']
    values_imp = [improvement_pp, improvement_ratio - 1]  # Show ratio - 1 for centered view
    colors_imp = ['red' if v < 0 else 'green' for v in values_imp]
    
    bars = ax.bar(metrics, values_imp, color=colors_imp, alpha=0.7, edgecolor='black', linewidth=2)
    ax.axhline(0, color='black', linestyle='-', linewidth=1)
    ax.set_ylabel('Value', fontsize=11, fontweight='bold')
    ax.set_title('Improvement Metrics', fontsize=12, fontweight='bold')
    
    # Add value labels
    ax.text(0, values_imp[0], f'{improvement_pp:+.2f}pp',
            ha='center', va='bottom' if values_imp[0] > 0 else 'top', fontsize=11, fontweight='bold')
    ax.text(1, values_imp[1], f'{improvement_ratio:.2f}x',
            ha='center', va='bottom' if values_imp[1] > 0 else 'top', fontsize=11, fontweight='bold')
    
    ax.grid(axis='y', alpha=0.3)
    
    fig.tight_layout()
    return save_and_log_plot(fig, "06_evaluation_summary", wandb_enabled)


def plot_action_kde_by_event(actions_traj, classified_event_array, var_names, wandb_enabled=False):
    """
    Plot kernel density estimates for each action variable, grouped by ENSO event type.
    
    Args:
        actions_traj (array): Actions trajectory (num_steps x num_actions)
        classified_event_array (array): Month-by-month event classifications
        var_names (list): Variable names (including Nino34)
        wandb_enabled (bool): Log to W&B
        
    Returns:
        str: Path to saved plot
    """
    action_vars = var_names[1:]  # Skip Nino34
    num_actions = len(action_vars)
    
    # Event categories and colors
    event_types = ['Multi-year El Nino', 'Multi-year La Nina', 'Single-year El Nino', 'Single-year La Nina', 'Neutral']
    colors = {'Multi-year El Nino': '#FF6347', 'Multi-year La Nina': '#1E90FF', 
              'Single-year El Nino': '#F4A460', 'Single-year La Nina': '#4682B4', 
              'Neutral': '#95B8D1'}
    
    # Trim classified_event_array to match actions_traj length
    # (classified_event_array may be one element longer due to initial state)
    classified_trimmed = classified_event_array[:actions_traj.shape[0]]
    
    # Create subplots
    fig, axes = plt.subplots(nrows=num_actions, ncols=1, figsize=(14, num_actions * 3), sharex=False)
    if num_actions == 1:
        axes = [axes]
    
    fig.suptitle('Action Variables Distribution by ENSO Event Type', fontsize=14, fontweight='bold')
    
    # Prepare data for each action
    for i, action_name in enumerate(action_vars):
        ax = axes[i]
        action_data = actions_traj[:, i]
        
        # Plot KDE for each event type
        for event_type in event_types:
            mask = classified_trimmed == event_type
            if np.sum(mask) > 0:  # Only plot if data exists
                event_actions = action_data[mask]
                sns.kdeplot(data=event_actions, ax=ax, label=event_type, 
                           color=colors.get(event_type, '#000000'), linewidth=2, alpha=0.7)
        
        ax.set_title(f'{action_name}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Action Value')
        ax.set_ylabel('Density')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='best')
    
    fig.tight_layout()
    return save_and_log_plot(fig, "08_action_kde_by_event", wandb_enabled)


def plot_state_kde_by_event(states_traj, classified_event_array, var_names, wandb_enabled=False):
    """
    Plot kernel density estimates for each state variable, grouped by ENSO event type.
    
    Args:
        states_traj (array): States trajectory (num_steps x num_variables)
        classified_event_array (array): Month-by-month event classifications
        var_names (list): Variable names
        wandb_enabled (bool): Log to W&B
        
    Returns:
        str: Path to saved plot
    """
    num_vars = len(var_names)
    
    # Event categories and colors
    event_types = ['Multi-year El Nino', 'Multi-year La Nina', 'Single-year El Nino', 'Single-year La Nina', 'Neutral']
    colors = {'Multi-year El Nino': '#FF6347', 'Multi-year La Nina': '#1E90FF', 
              'Single-year El Nino': '#F4A460', 'Single-year La Nina': '#4682B4', 
              'Neutral': '#95B8D1'}
    
    # Trim classified_event_array to match states_traj length
    # (classified_event_array may be one element longer due to initial state)
    classified_trimmed = classified_event_array[:states_traj.shape[0]]
    
    # Create subplots
    fig, axes = plt.subplots(nrows=num_vars, ncols=1, figsize=(14, num_vars * 3), sharex=False)
    if num_vars == 1:
        axes = [axes]
    
    fig.suptitle('State Variables Distribution by ENSO Event Type', fontsize=14, fontweight='bold')
    
    # Prepare data for each state variable
    for i, var_name in enumerate(var_names):
        ax = axes[i]
        state_data = states_traj[:, i]
        
        # Plot KDE for each event type
        for event_type in event_types:
            mask = classified_trimmed == event_type
            if np.sum(mask) > 0:  # Only plot if data exists
                event_states = state_data[mask]
                sns.kdeplot(data=event_states, ax=ax, label=event_type, 
                           color=colors.get(event_type, '#000000'), linewidth=2, alpha=0.7)
        
        ax.set_title(f'{var_name}', fontsize=11, fontweight='bold')
        ax.set_xlabel('State Value')
        ax.set_ylabel('Density')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='best')
    
    fig.tight_layout()
    return save_and_log_plot(fig, "09_state_kde_by_event", wandb_enabled)
