"""
Shared constants and helper functions for the ENSO-RL analysis notebooks.

Imported by both:
  - notebooks/analysis.ipynb       (simulation / behavior figures, uses inference.npz)
  - notebooks/analysis_xai.ipynb   (driver-attribution figures, uses per-method npz)

Keeping these here avoids duplicating the setup logic across the two notebooks.
"""
import warnings

import numpy as np
from scipy.stats import t as t_dist

# Phase keys, display labels, and plot colors used across every figure.
PHASES       = ["total", "el_nino", "la_nina"]
PHASE_LABELS = {"total": "Total MYE", "el_nino": "Multi-year El Niño", "la_nina": "Multi-year La Niña"}
# One phase palette for the whole paper: El Nino is ALWAYS red, La Nina ALWAYS blue.
PHASE_COLORS = {"total": "#6A3D9A", "el_nino": "#C1524E", "la_nina": "#4878CF"}

# PHASE_COLORS: red and blue must mean El Nino and La Nina everywhere in the paper and
METHOD_COLORS  = {"counterfactual": "#4D4D4D", "shapley": "#E08214",
                  "interventional": "#01665E"}
SIGN_COLORS    = {"pos": "#E08214", "neg": "#01665E"}   # sustain / brake
NS_COLOR       = "#cccccc"                              # not significant, all figures
DIVERGING_CMAP = "BrBG"                                 # signed heatmaps (not RdBu)
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


def _fdr(pvals):
    """Benjamini-Hochberg FDR-adjusted p-values (q-values) for a whole test family.

    The driver figures run one t-test per (mode, phase) -- 9 x 3 = 27 per analysis --
    so raw p-values overstate significance across the family: at alpha=0.05 roughly one
    of 27 is expected to pass by chance alone. BH controls the expected proportion of
    false discoveries among the calls actually made, which is the right criterion here
    (we want a reliable driver *set*, not protection against a single false positive,
    which is what the far stricter Bonferroni gives).

    Pass every p-value in the family AT ONCE -- all phases together, not phase by phase.
    Correcting within a phase would leave the across-phase family uncontrolled.

    Args:
        pvals: array-like of raw p-values; shape is preserved in the return value.

    Returns:
        np.ndarray of q-values, same shape as `pvals`.
    """
    p = np.asarray(pvals, float)
    flat = p.ravel()
    n = flat.size
    order = np.argsort(flat)
    q = np.empty(n)
    # BH step-up: q_(i) = min_{j >= i} ( n/j * p_(j) ), enforced by a reverse cumulative
    # minimum so q stays monotone in p.
    q[order] = np.minimum.accumulate(
        (flat[order] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.clip(q, 0.0, 1.0).reshape(p.shape)


def _add_sig_labels(ax, x_positions, values, pvals, cis=None, offset_frac=0.05):
    """Annotate a bar chart with significance stars, clear of the error whiskers.

    Labels sit beyond the WHISKER TIP (value +/- ci), not the bar top. Placing them at
    the bar top -- as this did previously -- drops the star on its own error bar
    whenever the CI is wide, which is exactly the case where the reader most needs to
    see both. Pass `cis` whenever the bars are drawn with `yerr`; omitting it falls
    back to bar-top placement, so existing callers are unaffected.

    The y-limits are widened to fit whatever was drawn, so labels are never clipped.

    Args:
        ax: target axes.
        x_positions: bar x centres.
        values: bar heights, in the same order.
        pvals: p- (or q-) values used to pick the star, same order.
        cis: half-width of each error bar, same order. None -> treated as zeros.
        offset_frac: gap between whisker tip and label, as a fraction of the y-range.
    """
    values = np.asarray(values, float)
    cis = np.zeros_like(values) if cis is None else np.abs(np.asarray(cis, float))
    lo, hi = ax.get_ylim()
    offset = (hi - lo) * offset_frac
    ymin, ymax = lo, hi
    for xi, val, p, ci in zip(x_positions, values, pvals, cis):
        up = val >= 0
        tip = val + ci if up else val - ci      # outer end of the whisker
        y = tip + offset if up else tip - offset
        ax.text(xi, y, _sig_stars(p), ha="center", va="bottom" if up else "top",
                fontsize=9, fontweight="bold")
        ymax = max(ymax, y + offset)            # headroom for the glyph itself
        ymin = min(ymin, y - offset)
    ax.set_ylim(ymin, ymax)


def _mean_ci(arr):
    """Mean and 95% CI (t-distribution) across seeds."""
    n = len(arr)
    ci = t_dist.ppf(0.975, df=n - 1) * arr.std(ddof=1) / np.sqrt(n)
    return arr.mean(), ci


def _load_npz(path):
    """Load npz and return as a dict-like NpzFile (supports 'key in d.files')."""
    return np.load(path, allow_pickle=True)


def _median_iqr(arr, axis=0):
    """Median and the 25th/75th percentiles along `axis` (the IQR band).

    Used for ensemble learning curves: median across seeds with a shaded 25-75%
    band. Robust to the non-normal, outlier-prone spread of RL runs across seeds,
    per rliable's recommendation. NaNs (e.g. the first PPO update before the logger
    is populated) are ignored.
    """
    arr = np.asarray(arr, float)
    with warnings.catch_warnings():  # all-NaN column (metric absent in a run) -> NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(arr, axis=axis)
        q25, q75 = np.nanpercentile(arr, [25, 75], axis=axis)
    return median, q25, q75


def _bootstrap_ci(values, statistic=np.mean, n=10000, ci=95, seed=0):
    """Bootstrap point estimate and a two-sided CI for `statistic` over `values`.

    Resamples `values` with replacement `n` times. Used for the headline final-lift
    number (stratified across the seed ensemble). Returns (point, lo, hi).
    """
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n, len(values)))
    boot = statistic(values[idx], axis=1)
    lo, hi = np.percentile(boot, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return float(statistic(values)), float(lo), float(hi)
