"""
Spontaneous-MYE precursor composite (paper point 2.1) — agent-free internal validation.

This script removes the agent entirely and asks: what does the 9-mode state look
like in the months *preceding* multi-year ENSO events that arise SPONTANEOUSLY in a
free-running XRO emulator? Compositing those lead-up windows gives the emulator's own
"natural precursor pattern" — which we can then compare against the agent-discovered
drivers. If they match, the agent's drivers are what actually precedes spontaneous MYE
in the same dynamics (a clean, cheap interpretation check, no observations needed).

Method:
  * Free-run XRO (zero actions) over many long rollouts, capturing the full per-month
    state (Nino3.4 + 9 modes).
  * Detect spontaneous multi-year events the same way the classifier does: continuous
    runs of |Nino3.4| past threshold lasting > min_duration months. Each qualifying
    run's start index is an event onset.
  * For every onset with enough lead-in, extract the preceding `lead`-month window of
    the full state and composite (mean across events, with paired-free CIs).
  * Split by phase (El Nino / La Nina onsets) — the asymmetry feeds paper point 1.2.

Note: windows are taken as-is; if events cluster, a lead window may overlap a prior
event's active months. For a first composite this is acceptable; report event counts.

Usage:
    uv run scripts/analysis/precursor_composite.py
    uv run scripts/analysis/precursor_composite.py --n-runs 100 --months 2400 --lead 24
"""
import sys
import time
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats as sp_stats

repo_root = Path(__file__).parent.parent.parent
analysis_dir = Path(__file__).parent
# Import shared agent-free machinery from the sibling interventional script.
sys.path.insert(0, str(analysis_dir))
sys.path.insert(0, str(repo_root))

from interventional_xro import build_env, rollout  # noqa: E402
from utils.enso_classifier import _find_continuous_runs  # noqa: E402
from utils import suppress_warnings  # noqa: E402

PHASES = ['total', 'el_nino', 'la_nina']
PHASE_LABELS = {'total': 'All MYE', 'el_nino': 'Multi-year El Nino',
                'la_nina': 'Multi-year La Nina'}


def collect_events(env, var_names, state_std, n_runs, months, lead,
                   min_duration, master_seed):
    """Free-run XRO and gather pre-onset state windows for spontaneous MYEs.

    Returns:
        windows: dict phase -> array [n_events, lead, n_modes] of pre-onset state.
        seeds:   the seeds used (for provenance).
    """
    rng = np.random.default_rng(master_seed)
    seeds = [int(rng.integers(0, 2**31)) for _ in range(n_runs)]
    threshold = env.threshold

    # El Nino / La Nina collected separately; 'total' is their union.
    collected = {'el_nino': [], 'la_nina': []}

    for seed in seeds:
        # action_idxs=None -> pure free run (zero actions); capture full states.
        _, states, enso = rollout(env, months, seed, action_idxs=None,
                                   magnitude=0.0, mode='press',
                                   state_std=state_std, return_states=True)
        phase_binary = {
            'el_nino': (enso >= threshold).astype(int),
            'la_nina': (enso <= -threshold).astype(int),
        }
        for phase, binary in phase_binary.items():
            for start, _end, length in _find_continuous_runs(binary):
                # Same multi-year criterion as classify_enso_event: length > min_duration.
                if length > min_duration and start - lead >= 0:
                    collected[phase].append(states[start - lead:start])

    windows = {}
    for phase in ('el_nino', 'la_nina'):
        windows[phase] = (np.stack(collected[phase]) if collected[phase]
                          else np.empty((0, lead, len(var_names))))
    windows['total'] = (np.concatenate([windows['el_nino'], windows['la_nina']])
                        if (windows['el_nino'].size or windows['la_nina'].size)
                        else np.empty((0, lead, len(var_names))))
    return windows, seeds


def composite_stats(windows, lead):
    """Mean precursor trajectory and averaged pattern per phase, with 95% CIs.

    Returns dict phase -> {
        'n', 'traj_mean' [lead, n_modes], 'traj_ci' [lead, n_modes],
        'pattern_mean' [n_modes], 'pattern_ci' [n_modes]
    }.
    """
    out = {}
    for phase, w in windows.items():
        n = w.shape[0]
        if n == 0:
            out[phase] = {'n': 0}
            continue
        traj_mean = w.mean(axis=0)                      # [lead, n_modes]
        traj_sem = w.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(traj_mean)
        tcrit = sp_stats.t.ppf(0.975, df=n - 1) if n > 1 else 0.0
        # Per-event average over the lead window -> one pattern vector per event.
        per_event_pattern = w.mean(axis=1)              # [n_events, n_modes]
        pattern_mean = per_event_pattern.mean(axis=0)
        pattern_sem = (per_event_pattern.std(axis=0, ddof=1) / np.sqrt(n)
                       if n > 1 else np.zeros_like(pattern_mean))
        out[phase] = {
            'n': n,
            'traj_mean': traj_mean,
            'traj_ci': tcrit * traj_sem,
            'pattern_mean': pattern_mean,
            'pattern_ci': tcrit * pattern_sem,
        }
    return out


def _print_table(stats, var_names, lead):
    print(f"\n{'='*88}")
    print(f"SPONTANEOUS-MYE PRECURSOR COMPOSITE — mean state over {lead} months before onset")
    print('='*88)
    for phase in PHASES:
        s = stats[phase]
        print(f"\n[{PHASE_LABELS[phase]}]  N_events = {s.get('n', 0)}")
        if s.get('n', 0) == 0:
            print("  (no events collected)")
            continue
        print(f"  {'Mode':<10} {'precursor':>12} {'±CI':>10} {'Sig':>5}")
        print('  ' + '-'*40)
        for i, name in enumerate(var_names):
            mean = s['pattern_mean'][i]
            ci = s['pattern_ci'][i]
            sig = '*' if (ci > 0 and abs(mean) > ci) else 'ns'
            print(f"  {name:<10} {mean:>+12.4f} {ci:>10.4f} {sig:>5}")


def _plot(stats, var_names, lead, output_dir):
    """Precursor evolution: composite state of each driver mode over the lead window."""
    phases = [p for p in PHASES if stats[p].get('n', 0) > 0]
    if not phases:
        print("  No events to plot.")
        return
    drivers = list(range(1, len(var_names)))  # skip Nino3.4 (index 0, the target)
    x = np.arange(-lead, 0)
    fig, axes = plt.subplots(1, len(phases), figsize=(7 * len(phases), 6), squeeze=False)
    cmap = plt.get_cmap('tab10')
    for k, phase in enumerate(phases):
        ax = axes[0][k]
        s = stats[phase]
        for j, mi in enumerate(drivers):
            ax.plot(x, s['traj_mean'][:, mi], color=cmap(j % 10),
                    label=var_names[mi], lw=1.8)
            ax.fill_between(x, s['traj_mean'][:, mi] - s['traj_ci'][:, mi],
                            s['traj_mean'][:, mi] + s['traj_ci'][:, mi],
                            color=cmap(j % 10), alpha=0.15)
        ax.axhline(0, color='gray', ls='--', lw=0.8)
        ax.set_xlabel('Months before MYE onset')
        ax.set_ylabel('Composite mode anomaly')
        ax.set_title(f"{PHASE_LABELS[phase]}  (N={s['n']})")
        ax.grid(alpha=0.3)
        if k == len(phases) - 1:
            ax.legend(fontsize=8, ncol=2, loc='best')
    fig.suptitle('Spontaneous-MYE precursor composite (agent-free XRO)', fontsize=14)
    fig.tight_layout()
    out = output_dir / 'precursor_composite.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved {out}")


def main():
    parser = argparse.ArgumentParser(description="Spontaneous-MYE precursor composite (XRO, no agent)")
    parser.add_argument("--n-runs", type=int, default=50, help="Free-run rollouts to pool events over")
    parser.add_argument("--months", type=int, default=2400, help="Months per rollout")
    parser.add_argument("--lead", type=int, default=24, help="Pre-onset window length (months)")
    parser.add_argument("--min-duration", type=int, default=12,
                        help="Multi-year criterion: run length > this (months). Matches classifier.")
    parser.add_argument("--master-seed", type=int, default=42)
    parser.add_argument("--prefix", type=str, default="rl_model",
                        help="Namespace label (match the ensemble prefix for figure grouping)")
    args = parser.parse_args()

    suppress_warnings()
    output_dir = Path("plots") / args.prefix / "precursor_composite"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SPONTANEOUS-MYE PRECURSOR COMPOSITE (XRO, no policy)")
    print("=" * 70)
    start = time.time()

    env, var_names, state_std = build_env()
    print(f"  N_RUNS={args.n_runs} MONTHS={args.months} LEAD={args.lead} "
          f"MIN_DURATION={args.min_duration}")

    windows, seeds = collect_events(env, var_names, state_std, args.n_runs,
                                    args.months, args.lead, args.min_duration,
                                    args.master_seed)
    stats = composite_stats(windows, args.lead)
    _print_table(stats, var_names, args.lead)
    _plot(stats, var_names, args.lead, output_dir)

    # Save flat arrays (per phase) for later overlay against driver rankings.
    save = {
        'var_names': np.array(var_names),
        'lead': args.lead, 'min_duration': args.min_duration,
        'n_runs': args.n_runs, 'months': args.months,
        'seeds': np.array(seeds),
    }
    for phase in PHASES:
        s = stats[phase]
        save[f'{phase}_n'] = s.get('n', 0)
        if s.get('n', 0) > 0:
            save[f'{phase}_traj_mean'] = s['traj_mean']
            save[f'{phase}_traj_ci'] = s['traj_ci']
            save[f'{phase}_pattern_mean'] = s['pattern_mean']
            save[f'{phase}_pattern_ci'] = s['pattern_ci']
    np.savez(output_dir / 'precursor_composite.npz', **save)
    print(f"  Saved {output_dir / 'precursor_composite.npz'}")

    el = time.time() - start
    print(f"\n{'='*70}\nPRECURSOR COMPOSITE COMPLETE — {int(el//60)}m {int(el%60)}s\n{'='*70}")


if __name__ == "__main__":
    main()
