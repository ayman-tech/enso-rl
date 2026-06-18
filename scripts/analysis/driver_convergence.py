"""
Driver convergence figure (paper point 1.5 payoff).

Overlays the per-mode driver importance from THREE independent attributions and
checks whether they agree:
  1. Counterfactual zero-ablation ΔP(MYE)   (plots/<prefix>/counterfactual/counterfactual_ensemble.npz)
  2. Shapley value on mye_prob              (plots/<prefix>/shapley/shapley_ensemble.npz)
  3. Agent-free interventional ΔP(MYE)      (plots/interventional_xro/interventional_xro.npz)

Methods 1-2 are policy-grounded (ensemble of trained agents); method 3 is the
agent-free causal check. Convergence across all three — especially the
sustainers-vs-brakes split and the El Nino/La Nina asymmetry — is the
review-proof evidence.

Because the three use different units, importance is z-scored within each method
before overlay; rank agreement is reported via Spearman rho on the raw values.

Usage:
    uv run scripts/analysis/driver_convergence.py
    uv run scripts/analysis/driver_convergence.py --phase la_nina
"""
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr

repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from utils.results_io import save_csv  # noqa: E402

PHASES = ['total', 'el_nino', 'la_nina']
PHASE_LABELS = {'total': 'Total MYE', 'el_nino': 'Multi-year El Nino',
                'la_nina': 'Multi-year La Nina'}

def npz_paths(prefix):
    """Per-ensemble npz locations, namespaced under plots/<prefix>/."""
    base = Path("plots") / prefix
    return (base / "counterfactual" / "counterfactual_ensemble.npz",
            base / "shapley" / "shapley_ensemble.npz",
            base / "interventional_xro" / "interventional_xro.npz")


def _zscore(x):
    x = np.asarray(x, dtype=float)
    sd = x.std()
    return (x - x.mean()) / sd if sd > 1e-12 else x - x.mean()


def load_counterfactual(phase, cf_npz):
    """Returns {feature: value}. Note: counterfactual ablation ΔP is NEGATIVE for
    drivers (removing a driver lowers P(MYE)); negate so 'higher = stronger driver'
    to align sign with Shapley/interventional."""
    if not cf_npz.exists():
        return None
    d = np.load(cf_npz, allow_pickle=True)
    feats = [str(f) for f in d['features']]
    return dict(zip(feats, -d[f'mean_{phase}']))


def load_shapley(phase, sh_npz):
    if not sh_npz.exists():
        return None
    d = np.load(sh_npz, allow_pickle=True)
    feats = [str(f) for f in d['features']]
    return dict(zip(feats, d[f'mean_{phase}']))


def load_interventional(phase, iv_npz, sign='sustain'):
    """Agent-free ΔP(MYE) per target for a phase; value = ΔP (higher = adds MYE).

    sign='+'/'-': use that fixed perturbation direction.
    sign='sustain' (default): per target, take the direction that maximally
        *sustains* this phase, i.e. max(ΔP₊, ΔP₋). Each mode's sustaining
        direction is mode- and phase-specific (e.g. for multi-year La Niña the
        inter-basin modes IOB/TNA/IOD/ATL3 sustain with +, while WWV/PMM/SIOD/SASD
        sustain with −), so any single global sign mis-ranks half the modes. The
        sustaining-direction value is the quantity directly comparable to the
        counterfactual/Shapley sustainer strength, so it is the correct input for
        the convergence overlay and Spearman.
    """
    if not iv_npz.exists():
        return None
    d = np.load(iv_npz, allow_pickle=True)
    targets = [str(t) for t in d['targets']]
    signs = [str(s) for s in d['signs']]
    phases = [str(p) for p in d['phases']]
    means = d['mean']

    if sign in ('+', '-'):
        return {t: float(m) for t, s, p, m in zip(targets, signs, phases, means)
                if s == sign and p == phase}

    # 'sustain': per target, the larger ΔP across the available ± directions.
    by_target = {}
    for t, s, p, m in zip(targets, signs, phases, means):
        if p == phase:
            by_target.setdefault(t, []).append(float(m))
    return {t: max(vals) for t, vals in by_target.items()}


def main():
    ap = argparse.ArgumentParser(description="Driver convergence across 3 methods")
    ap.add_argument("--prefix", type=str, default="rl_model",
                    help="Ensemble prefix; reads plots/<prefix>/ and writes there")
    ap.add_argument("--phase", choices=PHASES + ['all'], default='all')
    ap.add_argument("--iv-sign", choices=['+', '-', 'sustain'], default='sustain',
                    help="Interventional direction: 'sustain' (default; per-mode "
                         "sustaining direction, max of ±) or a fixed '+'/'-'")
    args = ap.parse_args()

    cf_npz, sh_npz, iv_npz = npz_paths(args.prefix)
    out_dir = Path("plots") / args.prefix / "convergence"
    out_dir.mkdir(parents=True, exist_ok=True)
    phases = PHASES if args.phase == 'all' else [args.phase]

    spearman_rows, importance_rows = [], []
    for phase in phases:
        cf = load_counterfactual(phase, cf_npz)
        sh = load_shapley(phase, sh_npz)
        iv = load_interventional(phase, iv_npz, args.iv_sign)

        iv_label = {'+': '+ΔP', '-': '-ΔP', 'sustain': 'sustaining ΔP'}[args.iv_sign]
        methods = {'Counterfactual (-ΔP)': cf, 'Shapley': sh,
                   f'Interventional ({iv_label})': iv}
        avail = {k: v for k, v in methods.items() if v}
        if len(avail) < 2:
            print(f"[{phase}] need >=2 methods present; found {list(avail)}. "
                  f"Run the ensemble + interventional scripts first.")
            continue

        # Common features (modes only; interventional also has groups -> intersect)
        common = set.intersection(*[set(v.keys()) for v in avail.values()])
        # keep a stable mode order: counterfactual/shapley feature order if present
        ref = cf or sh
        feats = [f for f in ref.keys() if f in common]

        # z-scored overlay
        fig, ax = plt.subplots(figsize=(max(10, len(feats)), 6))
        x = np.arange(len(feats))
        width = 0.8 / len(avail)
        for k, (mname, mdict) in enumerate(avail.items()):
            raw = [mdict[f] for f in feats]
            vals = _zscore(raw)
            for f, rv, zv in zip(feats, raw, vals):
                importance_rows.append({'phase': phase, 'method': mname, 'mode': f,
                                        'raw': float(rv), 'zscore': float(zv)})
            ax.bar(x + k * width, vals, width, label=mname, edgecolor='black', linewidth=0.5)
        ax.axhline(0, color='gray', ls='--', lw=0.8)
        ax.set_xticks(x + width * (len(avail) - 1) / 2)
        ax.set_xticklabels(feats, rotation=45, ha='right')
        ax.set_ylabel('Driver importance (z-scored within method)')
        ax.set_title(f'Driver convergence — {PHASE_LABELS[phase]}\n'
                     f'(higher = stronger sustainer; lower = brake)')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        fig.tight_layout()
        out = out_dir / f'convergence_{phase}.png'
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[{phase}] saved {out}")

        # Spearman rank agreement (raw values, pairwise)
        names = list(avail.keys())
        print(f"  Rank agreement (Spearman rho), {phase}:")
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a = [avail[names[i]][f] for f in feats]
                b = [avail[names[j]][f] for f in feats]
                rho, pv = spearmanr(a, b)
                print(f"    {names[i]:<26} vs {names[j]:<26}: rho={rho:+.3f} (p={pv:.3f})")
                spearman_rows.append({'phase': phase, 'method_a': names[i],
                                      'method_b': names[j], 'spearman_rho': float(rho),
                                      'p': float(pv)})

    # Tidy CSVs alongside the figures.
    save_csv(out_dir / 'convergence_spearman.csv', spearman_rows)
    save_csv(out_dir / 'convergence_importance.csv', importance_rows)


if __name__ == "__main__":
    main()
