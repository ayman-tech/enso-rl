"""
Shared constants and helper functions for the ENSO-RL analysis notebooks.

Imported by both:
  - notebooks/analysis.ipynb       (simulation / behavior figures, uses inference.npz)
  - notebooks/analysis_xai.ipynb   (driver-attribution figures, uses per-method npz)

Keeping these here avoids duplicating the setup logic across the two notebooks.
"""
import numpy as np
from scipy.stats import t as t_dist

# Phase keys, display labels, and plot colors used across every figure.
PHASES       = ["total", "el_nino", "la_nina"]
PHASE_LABELS = {"total": "Total MYE", "el_nino": "Multi-year El Niño", "la_nina": "Multi-year La Niña"}
PHASE_COLORS = {"total": "#4878CF", "el_nino": "#D65F5F", "la_nina": "#5CB85C"}
XRO_MODES    = ["WWV", "NPMM", "SPMM", "IOB", "IOD", "SIOD", "TNA", "ATL3", "SASD"]
SPINUP       = 12  # months skipped for seasonality binning (fixed by XRO env)

# Int8 MYE-label masks (encoding from inference.py: 3 = Multi-year El Niño, 4 = La Niña)
PHASE_MASKS = {
    'total':   lambda l: l >= 3,
    'el_nino': lambda l: l == 3,
    'la_nina': lambda l: l == 4,
}


def _zscore(x):
    x = np.asarray(x, float)
    sd = x.std()
    return (x - x.mean()) / sd if sd > 1e-12 else x - x.mean()


def _sig_stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def _add_sig_labels(ax, x_positions, values, pvals, offset_frac=0.04):
    """Annotate bar chart with significance stars."""
    ylim = ax.get_ylim()
    offset = (ylim[1] - ylim[0]) * offset_frac
    for xi, val, p in zip(x_positions, values, pvals):
        star = _sig_stars(p)
        y = val + offset if val >= 0 else val - offset
        va = "bottom" if val >= 0 else "top"
        ax.text(xi, y, star, ha="center", va=va, fontsize=9, fontweight="bold")


def _mean_ci(arr):
    """Mean and 95% CI (t-distribution) across seeds."""
    n = len(arr)
    ci = t_dist.ppf(0.975, df=n - 1) * arr.std(ddof=1) / np.sqrt(n)
    return arr.mean(), ci


def _load_npz(path):
    """Load npz and return as a dict-like NpzFile (supports 'key in d.files')."""
    return np.load(path, allow_pickle=True)
