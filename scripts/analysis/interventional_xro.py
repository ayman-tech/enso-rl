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
import multiprocessing as mp
from multiprocessing import cpu_count
import numpy as np
from pathlib import Path
from scipy import stats as sp_stats

repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from config import EnvConfig
from utils import suppress_warnings
from utils.results_io import save_csv
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
    params = prepare_xro_parameters(model_xro, train_ds, var_names, config=cfg)
    params['threshold'] = cfg.threshold
    env = XROMultiYearEnv(params=params, train_ds=train_ds,
                          var_names=var_names, max_steps=cfg.max_steps)
    # Observed per-mode std (state units) for pulse magnitude scaling
    state_std = np.array([float(np.nanstd(train_ds[v].values)) for v in var_names])
    return env, var_names, state_std


def rollout(env, num_months, seed, action_idxs=None, magnitude=0.0,
            mode='press', state_std=None, return_states=False):
    """One free-running rollout, optionally perturbing target modes.

    Args:
        action_idxs: indices into the 9-D action space (mode = var_names[idx+1]).
                     None -> pure baseline (zero actions).
        magnitude: signed perturbation size. press: action units; pulse: sigma units.
        mode: 'press' (every step) or 'pulse' (t0 offset only).
        return_states: if True, also return the full per-month state trajectory
                       (shape [num_months+1, n_modes]) and the Nino3.4 history,
                       for downstream analyses such as precursor composites (2.1).
    Returns:
        dict phase fractions from mye_fraction_by_phase; or, if return_states,
        the tuple (fractions, states, enso_history).
    """
    obs, _ = env.reset(seed=seed)

    if mode == 'pulse' and action_idxs is not None and magnitude != 0.0:
        for ai in action_idxs:
            gi = ai + 1  # state index (0 = Nino34, not actuated)
            env.state[gi] += magnitude * (state_std[gi] if state_std is not None else 1.0)
        obs = env._get_obs()

    enso_history = [env.state[0]]
    states = [env.state.copy()] if return_states else None
    action_vec = np.zeros(env.action_space.shape, dtype=np.float32)
    if mode == 'press' and action_idxs is not None:
        for ai in action_idxs:
            action_vec[ai] = magnitude

    for _ in range(num_months):
        obs, _, _, _, _ = env.step(action_vec if mode == 'press' else
                                   np.zeros(env.action_space.shape, dtype=np.float32))
        enso_history.append(env.state[0])
        if return_states:
            states.append(env.state.copy())

    classified = classify_enso_event(enso_history, threshold=env.threshold)
    fractions = mye_fraction_by_phase(classified)
    if return_states:
        return fractions, np.asarray(states), np.asarray(enso_history)
    return fractions


# Worker-process cache: the agent-free env is rebuilt once per worker (via the
# Pool initializer) and reused across every (target, sign) task it handles.
_WORKER_ENV = {}


def _init_worker():
    """Pool initializer: build the XRO env once per worker process."""
    suppress_warnings()
    env, var_names, state_std = build_env()
    _WORKER_ENV['env'] = env
    _WORKER_ENV['var_names'] = var_names
    _WORKER_ENV['state_std'] = state_std


def _compute_target_sign(env, var_names, state_std, tname, modes, sgn,
                         n_runs, months, magnitude, mode, master_seed):
    """Paired ΔP(MYE) per phase for ONE (target, sign), across n_runs seeds.

    Seeds are regenerated from master_seed, so the result is identical whether
    this runs serially or in a worker, and independent of task order. Returns
    the list of per-phase row dicts for this (target, sign).
    """
    rng = np.random.default_rng(master_seed)
    seeds = [int(rng.integers(0, 2**31)) for _ in range(n_runs)]
    name_to_idx = {v: i for i, v in enumerate(var_names[1:])}
    idxs = [name_to_idx[m] for m in modes]

    deltas = {p: np.zeros(n_runs) for p in PHASES}
    for i, seed in enumerate(seeds):
        base = rollout(env, months, seed, None, 0.0, mode, state_std)
        pert = rollout(env, months, seed, idxs, sgn * magnitude, mode, state_std)
        for p in PHASES:
            deltas[p][i] = pert[p] - base[p]

    rows = []
    for p in PHASES:
        d = deltas[p]
        mean = float(d.mean())
        se = d.std(ddof=1) / np.sqrt(n_runs) if n_runs > 1 else 0.0
        ci = float(sp_stats.t.ppf(0.975, df=n_runs - 1) * se) if n_runs > 1 else 0.0
        pval = (1.0 if (n_runs < 2 or np.allclose(d, 0))
                else float(sp_stats.ttest_1samp(d, 0.0)[1]))
        rows.append({'target': tname, 'sign': '+' if sgn > 0 else '-',
                     'phase': p, 'mean': mean, 'ci': ci, 'p': pval})
    return rows


def _worker_target_sign(task):
    """Worker entry: compute one (target, sign) using the per-worker cached env."""
    tname, modes, sgn, n_runs, months, magnitude, mode, master_seed = task
    rows = _compute_target_sign(
        _WORKER_ENV['env'], _WORKER_ENV['var_names'], _WORKER_ENV['state_std'],
        tname, modes, sgn, n_runs, months, magnitude, mode, master_seed)
    print(f"  [worker] {tname} {'+' if sgn > 0 else '-'} done", flush=True)
    return tname, sgn, rows


def paired_delta(env, targets, var_names, n_runs, months, magnitude, mode,
                 direction, state_std, master_seed, n_workers=1):
    """For each target & direction, paired ΔP(MYE) per phase across n_runs seeds.

    (target, sign) combinations are independent and parallelized across processes
    when n_workers > 1 (each worker builds its own env via the Pool initializer).
    Rows are reassembled in deterministic (target, sign) order, so the output is
    identical to the serial path. Returns (rows, seeds).
    """
    signs = {'both': [1.0, -1.0], 'pos': [1.0], 'neg': [-1.0]}[direction]
    tasks = [(tname, modes, sgn, n_runs, months, magnitude, mode, master_seed)
             for tname, modes in targets for sgn in signs]

    if n_workers > 1:
        print(f"  Parallel across {len(tasks)} (target,sign) tasks with "
              f"{n_workers} workers", flush=True)
        with mp.Pool(processes=n_workers, initializer=_init_worker) as pool:
            results = pool.map(_worker_target_sign, tasks)
    else:
        print("  Serial (workers=1)", flush=True)
        results = [(t[0], t[2],
                    _compute_target_sign(env, var_names, state_std,
                                         t[0], t[1], t[2], n_runs, months,
                                         magnitude, mode, master_seed))
                   for t in tasks]

    # Reassemble rows in task (target, sign) order, independent of completion order.
    key_to_rows = {(tname, sgn): rows for tname, sgn, rows in results}
    rows = []
    for tname, _modes, sgn, *_ in tasks:
        rows.extend(key_to_rows[(tname, sgn)])

    # Seeds (provenance) regenerated identically to the per-task seeds.
    rng = np.random.default_rng(master_seed)
    seeds = [int(rng.integers(0, 2**31)) for _ in range(n_runs)]
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


def main():
    # Use 'spawn' to avoid fork-related issues (matches the other analyses).
    mp.set_start_method('spawn', force=True)

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
    parser.add_argument("--model", type=str, default="ensemble",
                        help="Namespace label (match the ensemble prefix)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel workers across (target,sign) tasks "
                             "(default: cpu_count - 2; 1 = serial)")
    args = parser.parse_args()

    suppress_warnings()
    output_dir = Path("plots") / args.model / "interventional_xro"
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

    n_signs = {'both': 2, 'pos': 1, 'neg': 1}[args.direction]
    n_tasks = len(targets) * n_signs
    n_workers = args.workers if args.workers else max(1, cpu_count() - 2)
    n_workers = min(n_workers, n_tasks)  # no more workers than tasks

    print(f"  N_RUNS={args.n_runs} MONTHS={args.months} MODE={args.mode} "
          f"MAG={args.magnitude} DIR={args.direction} WORKERS={n_workers}")
    print(f"  targets: {[t[0] for t in targets]}")

    rows, seeds = paired_delta(env, targets, var_names, args.n_runs, args.months,
                               args.magnitude, args.mode, args.direction,
                               state_std, args.master_seed, n_workers=n_workers)
    _print_table(rows, args.n_runs)

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

    # Tidy CSV alongside the npz (rows are already per target/sign/phase).
    csv_rows = [{'target': r['target'], 'sign': r['sign'], 'phase': r['phase'],
                 'mean_dP_MYE': r['mean'], 'ci95': r['ci'], 'p': r['p']} for r in rows]
    save_csv(output_dir / 'interventional_xro.csv', csv_rows)

    el = time.time() - start
    print(f"\n{'='*70}\nINTERVENTIONAL ANALYSIS COMPLETE — {int(el//60)}m {int(el%60)}s\n{'='*70}")


if __name__ == "__main__":
    main()
