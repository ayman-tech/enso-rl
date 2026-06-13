"""
Agent-free interventional analysis on XRO — the causal backbone (paper point 1.5).

All five XAI methods explain the *trained policy*. This script removes the agent
entirely: it perturbs climate modes DIRECTLY in the XRO emulator and measures the
change in multi-year-ENSO probability. It answers the reviewer's central question
— "are these natural precursors, or just levers your controller happened to use?"

For each target (single mode or physically-motivated group), we compare a
free-running baseline (zero actions) against a perturbed run, paired by seed
(identical start state + noise), and report ΔP(MYE) split by phase
(total / El Nino / La Nina) with paired-t CIs.

Perturbation modes:
  * press : sustain a fixed nudge to the target mode(s) every month (via the env's
            action channel, magnitude in action units, no policy).
  * pulse : offset the target mode(s) once at t0 (in observed-sigma units), then
            free-run.

Direction:
  * both  : test +mag and -mag (effects can be sign-asymmetric); default.
  * pos / neg : single direction.

Usage:
    uv run scripts/analysis/interventional_xro.py --n-runs 30 --months 1200
    uv run scripts/analysis/interventional_xro.py --mode pulse --magnitude 1.0 --direction both
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
sys.path.insert(0, str(repo_root))

from config import EnvConfig
from utils import suppress_warnings
from utils.data_processing import load_observational_data, prepare_xro_parameters
from utils.enso_classifier import classify_enso_event, mye_fraction_by_phase
from envs import XROMultiYearEnv
from XRO.core import XRO

PHASES = ['total', 'el_nino', 'la_nina']
PHASE_LABELS = {'total': 'Total MYE', 'el_nino': 'Multi-year El Nino',
                'la_nina': 'Multi-year La Nina'}

# Physically-motivated mode groups (roadmap 1.5). Names must match var_names[1:].
GROUPS = {
    'inter_basin': ['ATL3', 'TNA', 'IOB', 'SASD'],
    'off_equatorial': ['NPMM', 'SPMM'],
    'recharge': ['WWV'],
}


def build_env():
    """Construct the XRO env directly (no agent/model needed)."""
    cfg = EnvConfig()
    _, train_ds, var_names, bounds = load_observational_data(
        cfg.data_config['data_path'],
        cfg.data_config['train_start'],
        cfg.data_config['train_end'])
    model_xro = XRO()
    params = prepare_xro_parameters(model_xro, train_ds, var_names, bounds)
    params['threshold'] = cfg.threshold
    env = XROMultiYearEnv(params=params, train_ds=train_ds,
                          var_names=var_names, max_steps=cfg.max_steps)
    # Observed per-mode std (state units) for pulse magnitude scaling
    state_std = np.array([float(np.nanstd(train_ds[v].values)) for v in var_names])
    return env, var_names, state_std


def rollout(env, num_months, seed, action_idxs=None, magnitude=0.0,
            mode='press', state_std=None):
    """One free-running rollout, optionally perturbing target modes.

    Args:
        action_idxs: indices into the 9-D action space (mode = var_names[idx+1]).
                     None -> pure baseline (zero actions).
        magnitude: signed perturbation size. press: action units; pulse: sigma units.
        mode: 'press' (every step) or 'pulse' (t0 offset only).
    Returns:
        dict phase fractions from mye_fraction_by_phase.
    """
    obs, _ = env.reset(seed=seed)

    if mode == 'pulse' and action_idxs is not None and magnitude != 0.0:
        for ai in action_idxs:
            gi = ai + 1  # state index (0 = Nino34, not actuated)
            env.state[gi] += magnitude * (state_std[gi] if state_std is not None else 1.0)
        obs = env._get_obs()

    enso_history = [obs[0]]
    action_vec = np.zeros(env.action_space.shape, dtype=np.float32)
    if mode == 'press' and action_idxs is not None:
        for ai in action_idxs:
            action_vec[ai] = magnitude

    for _ in range(num_months):
        obs, _, _, _, _ = env.step(action_vec if mode == 'press' else
                                   np.zeros(env.action_space.shape, dtype=np.float32))
        enso_history.append(obs[0])

    classified = classify_enso_event(enso_history, threshold=env.threshold)
    return mye_fraction_by_phase(classified)


def paired_delta(env, targets, var_names, n_runs, months, magnitude, mode,
                 direction, state_std, master_seed):
    """For each target & direction, paired ΔP(MYE) per phase across n_runs seeds.

    Returns: list of dicts with per-(target,direction,phase) stats.
    """
    rng = np.random.default_rng(master_seed)
    seeds = [int(rng.integers(0, 2**31)) for _ in range(n_runs)]
    name_to_idx = {v: i for i, v in enumerate(var_names[1:])}

    signs = {'both': [1.0, -1.0], 'pos': [1.0], 'neg': [-1.0]}[direction]

    rows = []
    for tname, modes in targets:
        idxs = [name_to_idx[m] for m in modes]
        for sgn in signs:
            deltas = {p: np.zeros(n_runs) for p in PHASES}
            for i, seed in enumerate(seeds):
                base = rollout(env, months, seed, None, 0.0, mode, state_std)
                pert = rollout(env, months, seed, idxs, sgn * magnitude, mode, state_std)
                for p in PHASES:
                    deltas[p][i] = pert[p] - base[p]
            for p in PHASES:
                d = deltas[p]
                mean = float(d.mean())
                se = d.std(ddof=1) / np.sqrt(n_runs) if n_runs > 1 else 0.0
                ci = float(sp_stats.t.ppf(0.975, df=n_runs - 1) * se) if n_runs > 1 else 0.0
                pval = (1.0 if (n_runs < 2 or np.allclose(d, 0))
                        else float(sp_stats.ttest_1samp(d, 0.0)[1]))
                rows.append({'target': tname, 'sign': '+' if sgn > 0 else '-',
                             'phase': p, 'mean': mean, 'ci': ci, 'p': pval})
    return rows, seeds


def _print_table(rows, n_runs):
    print(f"\n{'='*96}")
    print(f"AGENT-FREE INTERVENTIONAL ANALYSIS — ΔP(MYE) (N={n_runs} paired runs)")
    print('='*96)
    print(f"{'Target':<16} {'Sgn':<4} {'Phase':<10} | {'ΔP(MYE)':>10} {'±CI':>9} {'p':>9} {'Sig':>5}")
    print('-'*96)
    for r in rows:
        sig = ("***" if r['p'] < 0.001 else "**" if r['p'] < 0.01
               else "*" if r['p'] < 0.05 else "ns")
        print(f"{r['target']:<16} {r['sign']:<4} {r['phase']:<10} | "
              f"{r['mean']:>+10.4f} {r['ci']:>9.4f} {r['p']:>9.4f} {sig:>5}")


def _plot(rows, output_dir, mode, direction):
    """Bar chart of ΔP(MYE) per target for the 'total' phase, one panel per sign."""
    signs = sorted({r['sign'] for r in rows})
    targets = list(dict.fromkeys(r['target'] for r in rows))
    fig, axes = plt.subplots(1, len(signs), figsize=(8 * len(signs), 6), squeeze=False)
    for k, sgn in enumerate(signs):
        ax = axes[0][k]
        sub = [r for r in rows if r['sign'] == sgn and r['phase'] == 'total']
        sub = sorted(sub, key=lambda r: r['mean'])
        names = [r['target'] for r in sub]
        vals = [r['mean'] for r in sub]
        cis = [r['ci'] for r in sub]
        colors = ['#D32F2F' if r['p'] < 0.05 and r['mean'] > 0
                  else '#1976D2' if r['p'] < 0.05 else '#999999' for r in sub]
        ax.barh(names, vals, xerr=cis, color=colors, edgecolor='black', capsize=4)
        ax.axvline(0, color='gray', ls='--', lw=0.8)
        ax.set_xlabel('ΔP(MYE), total')
        ax.set_title(f'{mode} {sgn} perturbation')
        ax.grid(axis='x', alpha=0.3)
    fig.suptitle(f'Agent-free interventional ΔP(MYE) — {mode}/{direction}', fontsize=14)
    fig.tight_layout()
    out = output_dir / f'interventional_xro_{mode}.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved {out}")


def main():
    parser = argparse.ArgumentParser(description="Agent-free interventional analysis on XRO")
    parser.add_argument("--n-runs", type=int, default=30, help="Paired seeds")
    parser.add_argument("--months", type=int, default=1200, help="Months per rollout")
    parser.add_argument("--mode", choices=["press", "pulse"], default="press")
    parser.add_argument("--magnitude", type=float, default=1.0,
                        help="press: action units; pulse: observed-sigma units")
    parser.add_argument("--direction", choices=["both", "pos", "neg"], default="both")
    parser.add_argument("--master-seed", type=int, default=42)
    parser.add_argument("--groups-only", action="store_true",
                        help="Only run the mode groups, skip singles")
    args = parser.parse_args()

    suppress_warnings()
    output_dir = Path("plots/interventional_xro")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("AGENT-FREE INTERVENTIONAL ANALYSIS (XRO, no policy)")
    print("=" * 70)
    start = time.time()

    env, var_names, state_std = build_env()
    modes = list(var_names[1:])

    # Targets: singles (each mode alone) + physically-motivated groups
    targets = []
    if not args.groups_only:
        targets += [(m, [m]) for m in modes]
    targets += [(gname, gmodes) for gname, gmodes in GROUPS.items()]

    print(f"  N_RUNS={args.n_runs} MONTHS={args.months} MODE={args.mode} "
          f"MAG={args.magnitude} DIR={args.direction}")
    print(f"  targets: {[t[0] for t in targets]}")

    rows, seeds = paired_delta(env, targets, var_names, args.n_runs, args.months,
                               args.magnitude, args.mode, args.direction,
                               state_std, args.master_seed)
    _print_table(rows, args.n_runs)
    _plot(rows, output_dir, args.mode, args.direction)

    # Save flat arrays for the convergence figure (Part D)
    np.savez(
        output_dir / 'interventional_xro.npz',
        targets=np.array([r['target'] for r in rows]),
        signs=np.array([r['sign'] for r in rows]),
        phases=np.array([r['phase'] for r in rows]),
        mean=np.array([r['mean'] for r in rows]),
        ci=np.array([r['ci'] for r in rows]),
        p=np.array([r['p'] for r in rows]),
        mode=args.mode, direction=args.direction, magnitude=args.magnitude,
        n_runs=args.n_runs, months=args.months, seeds=np.array(seeds),
    )
    print(f"  Saved {output_dir / 'interventional_xro.npz'}")

    el = time.time() - start
    print(f"\n{'='*70}\nINTERVENTIONAL ANALYSIS COMPLETE — {int(el//60)}m {int(el%60)}s\n{'='*70}")


if __name__ == "__main__":
    main()
