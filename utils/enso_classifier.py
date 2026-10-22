"""
ENSO event classification utilities.
"""
import numpy as np

# Analysis-side ENSO index convention. ONI is a 3-month running mean of Nino3.4,
# and the conventional event definition is built on it, so every *reported* event
# statistic smooths before thresholding. The environment deliberately does NOT
# import this: the XRO state evolves on monthly anomalies and the reward keys on
# the raw monthly value, mirroring the XRO paper, which fits on monthly anomalies
# but reports skill on the 3-month running mean (Zhao et al., Nature 630, 2024).
ONI_SMOOTH_MONTHS = 3


def running_mean(x, window=3, min_periods=1):
    """Centered running mean for retrospective event analysis.

    An odd window is required: window=3 averages t-1, t, t+1. At the
    record boundaries, average available samples when min_periods permits;
    min_periods=window requires a complete window. Training rewards use raw
    monthly states and do not call this function.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"running_mean expects a 1-D series, got shape {x.shape}")
    if not isinstance(window, (int, np.integer)) or window < 1 or window % 2 == 0:
        raise ValueError("window must be a positive odd integer")
    if not 1 <= min_periods <= window:
        raise ValueError("min_periods must lie between 1 and window")
    if window == 1:
        return x.copy()
    half = window // 2
    indices = np.arange(len(x))
    lo = np.maximum(indices - half, 0)
    hi = np.minimum(indices + half + 1, len(x))
    csum = np.concatenate(([0.0], np.cumsum(x)))
    counts = hi - lo
    out = (csum[hi] - csum[lo]) / counts
    out[counts < min_periods] = np.nan
    return out


def classify_enso_event(enso_history, threshold=0.5, min_duration=12,
                        smooth_window=1):
    """
    Classifies each month in ENSO history into different event types.

    Args:
        enso_history (list or np.ndarray): List/array of ENSO index values over time
        threshold (float): Magnitude threshold for El Nino/La Nina
        min_duration (int): Minimum months for multi-year event (>min_duration = multi-year)
        smooth_window (int): Centered running-mean window applied to
            ``enso_history`` BEFORE thresholding. The default 1 is the identity,
            so existing callers are unaffected; analysis callers that want the
            ONI convention pass ``ONI_SMOOTH_MONTHS``. Smoothing bridges
            single-month dips, so runs merge and brief excursions are clipped --
            it is not a monotone change to event counts.

    Returns:
        np.ndarray: Array of classifications for each month
                   ('Neutral', 'Single-year El Nino', 'Single-year La Nina',
                    'Multi-year El Nino', 'Multi-year La Nina')
    """
    enso_history = np.array(enso_history)
    n_months = len(enso_history)
    if smooth_window > 1:
        enso_history = running_mean(enso_history, smooth_window)

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


def mye_fraction_by_phase(classified):
    """Fraction of months in a multi-year ENSO event, split by phase.

    Single source of truth for phase-resolved mye_prob, used by the lift analysis
    and the driver-attribution scripts so every result uses the same definition.

    Args:
        classified (np.ndarray): Output of classify_enso_event (per-month labels).

    Returns:
        dict: {'total', 'el_nino', 'la_nina'} fractions of len(classified).
    """
    classified = np.asarray(classified)
    n = len(classified)
    if n == 0:
        return {'total': 0.0, 'el_nino': 0.0, 'la_nina': 0.0}
    el_nino = int(np.sum(classified == 'Multi-year El Nino'))
    la_nina = int(np.sum(classified == 'Multi-year La Nina'))
    return {
        'total': (el_nino + la_nina) / n,
        'el_nino': el_nino / n,
        'la_nina': la_nina / n,
    }


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
