"""
ENSO event classification utilities.
"""
import numpy as np


def max_run_length(binary_sequence):
    """
    Calculates the longest duration of consecutive 1s in a binary sequence.
    
    Args:
        binary_sequence (np.ndarray): Binary array of 0s and 1s
        
    Returns:
        int: Maximum consecutive run length
    """
    padded = np.concatenate(([0], binary_sequence, [0]))
    changes = np.diff(padded)
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    
    if len(starts) == 0:
        return 0
    
    return (ends - starts).max()


def classify_enso_event(enso_history, threshold=0.5, min_duration=12):
    """
    Classifies each month in ENSO history into different event types.

    Args:
        enso_history (list or np.ndarray): List/array of ENSO index values over time
        threshold (float): Magnitude threshold for El Nino/La Nina
        min_duration (int): Minimum months for multi-year event (>min_duration = multi-year)

    Returns:
        np.ndarray: Array of classifications for each month
                   ('Neutral', 'Single-year El Nino', 'Single-year La Nina',
                    'Multi-year El Nino', 'Multi-year La Nina')
    """
    enso_history = np.array(enso_history)
    n_months = len(enso_history)

    # Initialize all as neutral
    classifications = np.array(['Neutral'] * n_months, dtype=object)

    # Identify El Niño and La Niña periods
    is_nino = (enso_history >= threshold).astype(int)
    is_nina = (enso_history <= -threshold).astype(int)

    # Find continuous runs for El Niño
    nino_runs = _find_continuous_runs(is_nino)
    for start, end, length in nino_runs:
        if length > min_duration:
            classifications[start:end+1] = 'Multi-year El Nino'
        elif length >= 6:
            classifications[start:end+1] = 'Single-year El Nino'

    # Find continuous runs for La Niña
    nina_runs = _find_continuous_runs(is_nina)
    for start, end, length in nina_runs:
        if length > min_duration:
            classifications[start:end+1] = 'Multi-year La Nina'
        elif length >= 6:
            classifications[start:end+1] = 'Single-year La Nina'

    return classifications


def _find_continuous_runs(binary_sequence):
    """
    Find all continuous runs of 1s in a binary sequence.
    
    Args:
        binary_sequence (np.ndarray): Binary array of 0s and 1s
        
    Returns:
        list: List of tuples (start_idx, end_idx, length) for each run of 1s
    """
    padded = np.concatenate(([0], binary_sequence, [0]))
    changes = np.diff(padded)
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0] - 1
    
    runs = []
    for start, end in zip(starts, ends):
        runs.append((start, end, end - start + 1))
    
    return runs


def summarize_classification(classifications_array):
    """
    Create a summary string showing percentages of all event types.
    
    Args:
        classifications_array (np.ndarray): Array of classifications for each month
        
    Returns:
        str: Summary with percentages (e.g., "Multi-year El Nino: 45.5%, Multi-year La Nina: 20.3%, ...")
    """
    from collections import Counter
    
    total_months = len(classifications_array)
    if total_months == 0:
        return 'No data'
    
    # Count all event types
    counts = Counter(classifications_array)
    
    # Calculate percentages for all events, sorted by count descending
    event_percentages = []
    for event, count in sorted(counts.items(), key=lambda x: -x[1]):
        percentage = (count / total_months) * 100
        event_percentages.append(f"{event}: {percentage:.1f}%")
    
    return ", ".join(event_percentages)
