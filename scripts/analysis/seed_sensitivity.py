"""
Seed-sensitivity / robustness-to-randomness displays (advisor request).

Two views that show the headline findings are stable across the 10 trained seeds,
not just on the ensemble average:

  asymmetry : per-seed scatter of El Nino vs La Nina lift. If every seed sits above
              the y=x line, the "La Nina is easier to sustain" asymmetry is unanimous,
              not an averaging artifact.
  ranking   : seed x mode heatmap of per-seed driver sustainer-strength (counterfactual
              or Shapley). Consistent columns down the seeds = a seed-stable ranking.

Data sources (written by the ensemble analyses):
  lift_ensemble.npz  -> per_seed_lift_{el_nino,la_nina}  (else parse a lift_*.out log)
  {counterfactual,shapley}_ensemble.npz -> per_seed_{total,el_nino,la_nina}

Usage:
    uv run scripts/analysis/seed_sensitivity.py --prefix ensemble
    uv run scripts/analysis/seed_sensitivity.py --prefix ensemble --plot asymmetry --lift-log logs/lift_7006985.out
    uv run scripts/analysis/seed_sensitivity.py --prefix ensemble --plot ranking --method shapley
"""
import re
import sys
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.results_io import save_csv  # noqa: E402

PHASES = ['total', 'el_nino', 'la_nina']
PHASE_LABELS = {'total': 'Total MYE', 'el_nino': 'Multi-year El Nino',
                'la_nina': 'Multi-year La Nina'}


def load_lift_per_seed(prefix, lift_log):
    """Return (seeds, en_lift, ln_lift, source). Prefer the npz; else parse a log."""
    npz = Path('plots') / prefix / 'lift' / 'lift_ensemble.npz'
    if npz.exists():
        d = np.load(npz, allow_pickle=True)
        if 'per_seed_lift_el_nino' in d.files:
            return (d['seeds'], d['per_seed_lift_el_nino'],
                    d['per_seed_lift_la_nina'], str(npz))
    # Fallback: parse a lift analysis log for the per-seed lines.
    if lift_log is None:
        cands = sorted(glob.glob('logs/lift_*.out'))
        lift_log = cands[-1] if cands else None
    if not lift_log or not Path(lift_log).exists():
        return None
    txt = Path(lift_log).read_text()
    rows = re.findall(
        r'seed=(\d+): lift total=([+\-][\d.]+) \| EN=([+\-][\d.]+) \| LN=([+\-][\d.]+)', txt)
    if not rows:
        return None
    seeds = np.array([int(r[0]) for r in rows])
    en = np.array([float(r[2]) for r in rows])
    ln = np.array([float(r[3]) for r in rows])
    return seeds, en, ln, lift_log


def plot_asymmetry(prefix, lift_log, out_dir):
    loaded = load_lift_per_seed(prefix, lift_log)
    if loaded is None:
        print("[asymmetry] no per-seed lift found (need lift_ensemble.npz or logs/lift_*.out)")
        return
    seeds, en, ln, src = loaded
    n = len(seeds)
    n_ln_gt_en = int(np.sum(ln > en))

    lim = float(max(0.01, en.max(), ln.max(), -en.min(), -ln.min()) * 1.1)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([-lim, lim], [-lim, lim], ls='--', color='gray', lw=1,
            label='La Nina = El Nino')
    ax.fill_between([-lim, lim], [-lim, lim], [lim, lim], color='#E8F5E9', zorder=0)
    ax.scatter(en, ln, s=90, c='#2E7D32', edgecolor='black', zorder=3)
    for i, s in enumerate(seeds):
        ax.annotate(str(int(s)), (en[i], ln[i]), fontsize=8,
                    xytext=(4, 4), textcoords='offset points')
    ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal')
    ax.set_xlabel('Per-seed El Nino lift  ΔP(MYE)')
    ax.set_ylabel('Per-seed La Nina lift  ΔP(MYE)')
    ax.set_title(f'Phase-asymmetry stability across seeds\n'
                 f'{n_ln_gt_en}/{n} seeds: La Nina lift > El Nino lift '
                 f'(points above the diagonal)')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = out_dir / 'seed_asymmetry.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[asymmetry] {n_ln_gt_en}/{n} seeds have La Nina > El Nino lift  (source: {src})")
    print(f"[asymmetry] saved {out}")
    save_csv(out_dir / 'seed_asymmetry.csv',
             [{'seed': int(seeds[i]), 'el_nino_lift': float(en[i]),
               'la_nina_lift': float(ln[i]), 'la_nina_gt_el_nino': bool(ln[i] > en[i])}
              for i in range(n)])


def load_per_seed_driver(prefix, method):
    """Return (features, seeds, {phase: [n_seeds, n_feat]}, source) or a status string."""
    npz = Path('plots') / prefix / method / f'{method}_ensemble.npz'
    if not npz.exists():
        return f'MISSING:{npz}'
    d = np.load(npz, allow_pickle=True)
    if 'per_seed_total' not in d.files:
        return f'NO_PER_SEED:{npz}'  # produced before per-seed saving was added
    feats = [str(f) for f in d['features']]
    return feats, d['seeds'], {p: d[f'per_seed_{p}'] for p in PHASES}, str(npz)


def plot_ranking(prefix, method, out_dir):
    loaded = load_per_seed_driver(prefix, method)
    if isinstance(loaded, str):
        if loaded.startswith('NO_PER_SEED'):
            print(f"[ranking] {method}: npz has no per-seed arrays yet — re-run "
                  f"`make {method} name={prefix}` after the per-seed-saving update.")
        else:
            print(f"[ranking] {method}: file not found ({loaded.split(':',1)[1]}).")
        return
    feats, seeds, per_seed, src = loaded
    # Common "sustainer strength" sign: counterfactual ΔP is negative for sustainers.
    sign = -1.0 if method == 'counterfactual' else 1.0

    fig, axes = plt.subplots(1, len(PHASES), figsize=(5.5 * len(PHASES), 5), squeeze=False)
    for k, phase in enumerate(PHASES):
        ax = axes[0][k]
        M = sign * np.asarray(per_seed[phase])          # [n_seeds, n_feat], higher=sustainer
        order = np.argsort(M.mean(axis=0))[::-1]        # sort modes by mean sustainer strength
        Msorted = M[:, order]
        names = [feats[i] for i in order]
        vmax = float(np.abs(M).max()) or 1.0
        im = ax.imshow(Msorted, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(len(seeds)))
        ax.set_yticklabels([f'seed {int(s)}' for s in seeds], fontsize=8)
        ax.set_title(f'{PHASE_LABELS[phase]}')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label='sustainer strength')
    fig.suptitle(f'Per-seed driver ranking stability — {method} '
                 f'(red = sustainer; columns sorted by ensemble mean)', fontsize=13)
    fig.tight_layout()
    out = out_dir / f'seed_ranking_{method}.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[ranking] saved {out}  (source: {src})")


def main():
    ap = argparse.ArgumentParser(description="Seed-sensitivity / robustness-to-randomness plots")
    ap.add_argument("--prefix", type=str, default="ensemble", help="Ensemble prefix")
    ap.add_argument("--plot", choices=['asymmetry', 'ranking', 'both'], default='both')
    ap.add_argument("--method", choices=['counterfactual', 'shapley', 'both'],
                    default='both', help="Which driver method for the ranking heatmap")
    ap.add_argument("--lift-log", type=str, default=None,
                    help="Lift log to parse if lift_ensemble.npz is absent")
    args = ap.parse_args()

    out_dir = Path('plots') / args.prefix / 'seed_sensitivity'
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.plot in ('asymmetry', 'both'):
        plot_asymmetry(args.prefix, args.lift_log, out_dir)
    if args.plot in ('ranking', 'both'):
        methods = ['counterfactual', 'shapley'] if args.method == 'both' else [args.method]
        for m in methods:
            plot_ranking(args.prefix, m, out_dir)


if __name__ == "__main__":
    main()
