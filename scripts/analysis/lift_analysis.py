"""
Multi-year-ENSO probability LIFT analysis, phase-resolved (paper points 1.1-1.3).

Reports the headline result of the paper: how much the RL agent raises the
probability of multi-year ENSO over a free-running (zero-action) baseline, with
confidence intervals, split into El Nino vs La Nina (the multi-year-La-Nina
asymmetry).

Runs in ensemble mode (CIs across independently trained seeds). Results are
saved to lift_ensemble.npz and lift_ensemble.csv for notebook plotting.

Usage:
    uv run scripts/analysis/lift_analysis.py --model ensemble \
        --seeds 0 1 2 3 4 5 6 7 8 9 --n-rollouts 30 --months 1200
"""
import sys
import time
import argparse
import numpy as np
import wandb
from pathlib import Path
from datetime import datetime
from scipy import stats as sp_stats

repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from utils import suppress_warnings
from utils.results_io import save_csv
from utils.evaluation import rollout_mye_phased, rollout_mye_events
from config import EnvConfig
# NOTE: load_environment is imported lazily inside run_ensemble to
# avoid a circular import (scripts.evaluate imports run_lift_evaluation from here).

PHASES = ['total', 'el_nino', 'la_nina']
PHASE_LABELS = {'total': 'Total MYE', 'el_nino': 'Multi-year El Nino',
                'la_nina': 'Multi-year La Nina'}


def _ci95(samples):
    """Mean and 95% t-CI half-width for a 1D sample array."""
    samples = np.asarray(samples, dtype=float)
    n = len(samples)
    mean = float(samples.mean())
    if n < 2:
        return mean, 0.0
    se = samples.std(ddof=1) / np.sqrt(n)
    ci = float(sp_stats.t.ppf(0.975, df=n - 1) * se)
    return mean, ci


def paired_lift_for_model(model, env, n_rollouts, months, master_seed=42):
    """Run n_rollouts paired (agent, baseline) rollouts for one model.

    Returns per phase: the time-fraction arrays (agent/baseline/lift) AND the
    decomposition ingredients — per-rollout multi-year event COUNTS and pooled
    per-event DURATIONS, for agent and baseline (so the lift can be split into
    'more events' vs 'longer events').
    """
    rng = np.random.default_rng(master_seed)
    seeds = [int(rng.integers(0, 2**31)) for _ in range(n_rollouts)]

    agent = {p: np.zeros(n_rollouts) for p in PHASES}
    base = {p: np.zeros(n_rollouts) for p in PHASES}
    agent_count = {p: np.zeros(n_rollouts) for p in PHASES}
    base_count = {p: np.zeros(n_rollouts) for p in PHASES}
    agent_dur = {p: [] for p in PHASES}
    base_dur = {p: [] for p in PHASES}

    for i, seed in enumerate(seeds):
        a = rollout_mye_events(env, agent=model, num_months=months, seed=seed)
        b = rollout_mye_events(env, agent=None, num_months=months, seed=seed)
        for p in PHASES:
            agent[p][i] = a[p]['frac']
            base[p][i] = b[p]['frac']
            agent_count[p][i] = a[p]['count']
            base_count[p][i] = b[p]['count']
            agent_dur[p].extend(a[p]['durations'])
            base_dur[p].extend(b[p]['durations'])

    lift = {p: agent[p] - base[p] for p in PHASES}
    return {'agent': agent, 'baseline': base, 'lift': lift, 'seeds': seeds,
            'agent_count': agent_count, 'base_count': base_count,
            'agent_dur': agent_dur, 'base_dur': base_dur}


def _print_table(title, rows):
    """rows: list of (label, agent_m, agent_ci, base_m, base_ci, lift_m, lift_ci, p)."""
    print(f"\n{'='*92}")
    print(title)
    print('='*92)
    print(f"{'Phase':<22} | {'Agent':>16} | {'Baseline':>16} | {'Lift':>16} | {'p':>8}")
    print('-'*92)
    for label, am, aci, bm, bci, lm, lci, p in rows:
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"{label:<22} | {am:>7.3f} ±{aci:<7.3f} | {bm:>7.3f} ±{bci:<7.3f} | "
              f"{lm:>+7.3f} ±{lci:<6.3f} | {p:>6.4f} {sig}")


def _mean_ci_nan(vals):
    """Mean and 95% t-CI over a list, ignoring NaNs (seeds with no events)."""
    a = np.asarray([v for v in vals if not np.isnan(v)], dtype=float)
    if a.size == 0:
        return float('nan'), 0.0
    if a.size < 2:
        return float(a[0]), 0.0
    return float(a.mean()), float(sp_stats.t.ppf(0.975, a.size - 1) * a.std(ddof=1) / np.sqrt(a.size))


def _print_decomp_table(decomp_rows):
    print(f"\n{'='*92}")
    print("LIFT DECOMPOSITION — multi-year event FREQUENCY (per 100 yr) and mean DURATION (mo)")
    print('='*92)
    print(f"{'Phase':<22} | {'Freq agent':>12} {'Freq base':>12} | {'Dur agent':>12} {'Dur base':>12}")
    print('-'*92)
    for r in decomp_rows:
        print(f"{PHASE_LABELS[r['phase']]:<22} | "
              f"{r['freq_agent_per100yr']:>7.2f}±{r['freq_agent_ci95']:<4.2f} "
              f"{r['freq_base_per100yr']:>7.2f}±{r['freq_base_ci95']:<4.2f} | "
              f"{r['dur_agent_mo']:>7.2f}±{r['dur_agent_ci95']:<4.2f} "
              f"{r['dur_base_mo']:>7.2f}±{r['dur_base_ci95']:<4.2f}")


def _log_lift_wandb(rows):
    """Log lift rows into the CURRENTLY ACTIVE wandb run.

    Does NOT call wandb.init/finish — so when invoked from evaluate.py the lift
    metrics land under evaluate.py's run. No-op if no run is active.
    """
    if wandb.run is None:
        return
    table = wandb.Table(
        columns=["Phase", "Agent", "Agent CI", "Baseline", "Baseline CI", "Lift", "Lift CI", "p"],
        data=[[PHASE_LABELS[p], am, aci, bm, bci, lm, lci, pv]
              for (p, am, aci, bm, bci, lm, lci, pv) in rows])
    log = {"lift_summary": table}
    for (p, am, aci, bm, bci, lm, lci, pv) in rows:
        log[f"lift/{p}"] = lm
        log[f"lift/{p}_ci"] = lci
    wandb.log(log)


def run_lift_evaluation(model, env, n_rollouts=100, months=1200, master_seed=42,
                        output_dir=None, label="model", wandb_log=False):
    """Phase-resolved MYE lift for an ALREADY-LOADED model + env (points 1.1, 1.2).

    Reusable entry point so evaluate.py can call lift without reloading the model.
    Returns the rows list: (phase, agent_m, agent_ci, base_m, base_ci, lift_m, lift_ci, p).
    """
    if output_dir is None:
        output_dir = Path("plots/lift")
    output_dir.mkdir(parents=True, exist_ok=True)

    res = paired_lift_for_model(model, env, n_rollouts, months, master_seed)

    rows = []
    for p in PHASES:
        am, aci = _ci95(res['agent'][p])
        bm, bci = _ci95(res['baseline'][p])
        lm, lci = _ci95(res['lift'][p])
        lift = res['lift'][p]  # paired t-test on the lift (H0: lift == 0)
        pval = 1.0 if np.allclose(lift, 0) else float(sp_stats.ttest_1samp(lift, 0.0)[1])
        rows.append((p, am, aci, bm, bci, lm, lci, pval))

    _print_table(f"MYE LIFT — '{label}' (N={n_rollouts} paired rollouts, {months} mo)", rows)
    save_kw = {'phases': np.array(PHASES), 'n_rollouts': n_rollouts, 'months': months}
    for p in PHASES:
        save_kw[f'agent_{p}'] = res['agent'][p]
        save_kw[f'baseline_{p}'] = res['baseline'][p]
        save_kw[f'lift_{p}'] = res['lift'][p]
    np.savez(output_dir / 'lift_single.npz', **save_kw)

    if wandb_log:
        _log_lift_wandb(rows)
    return rows


def run_ensemble(args, output_dir):
    from scripts.evaluate import load_environment  # lazy: avoid circular import
    env_config = EnvConfig()
    # Per-seed mean lift -> CIs computed ACROSS seeds (point 1.3)
    per_seed_lift = {p: [] for p in PHASES}
    per_seed_agent = {p: [] for p in PHASES}
    per_seed_base = {p: [] for p in PHASES}
    # Lift decomposition: per-seed event FREQUENCY (per 100 yr) and mean DURATION (mo)
    per_seed_freq_a = {p: [] for p in PHASES}
    per_seed_freq_b = {p: [] for p in PHASES}
    per_seed_dur_a = {p: [] for p in PHASES}
    per_seed_dur_b = {p: [] for p in PHASES}
    used_seeds = []
    cent = 1200.0 / args.months  # events-per-rollout -> events per 100 yr

    for seed in args.seeds:
        name = f"{args.model}_seed{seed}"
        try:
            model, env, _ = load_environment(name, env_config)
        except FileNotFoundError:
            print(f"  [skip] model not found: models/{name}.zip")
            continue
        res = paired_lift_for_model(model, env, args.n_rollouts, args.months, args.master_seed)
        for p in PHASES:
            per_seed_lift[p].append(float(res['lift'][p].mean()))
            per_seed_agent[p].append(float(res['agent'][p].mean()))
            per_seed_base[p].append(float(res['baseline'][p].mean()))
            per_seed_freq_a[p].append(float(np.mean(res['agent_count'][p])) * cent)
            per_seed_freq_b[p].append(float(np.mean(res['base_count'][p])) * cent)
            per_seed_dur_a[p].append(float(np.mean(res['agent_dur'][p])) if res['agent_dur'][p] else np.nan)
            per_seed_dur_b[p].append(float(np.mean(res['base_dur'][p])) if res['base_dur'][p] else np.nan)
        used_seeds.append(seed)
        print(f"  seed={seed}: lift total={per_seed_lift['total'][-1]:+.3f} "
              f"| EN={per_seed_lift['el_nino'][-1]:+.3f} "
              f"| LN={per_seed_lift['la_nina'][-1]:+.3f}")

    if not used_seeds:
        raise FileNotFoundError("No ensemble models found. Train with scripts/train_ensemble.py first.")

    rows = []
    for p in PHASES:
        am, aci = _ci95(per_seed_agent[p])
        bm, bci = _ci95(per_seed_base[p])
        lm, lci = _ci95(per_seed_lift[p])
        lift = np.asarray(per_seed_lift[p])
        if len(lift) < 2 or np.allclose(lift, 0):
            pval = 1.0
        else:
            _, pval = sp_stats.ttest_1samp(lift, 0.0)
        rows.append((p, am, aci, bm, bci, lm, lci, float(pval)))

    _print_table(f"MYE LIFT — ensemble '{args.model}' "
                 f"(N={len(used_seeds)} seeds x {args.n_rollouts} rollouts)", rows)

    # Cross-seed spread (strategy degeneracy signal)
    print(f"\n  Cross-seed spread (std of per-seed mean lift):")
    for p in PHASES:
        print(f"    {PHASE_LABELS[p]:<22}: std = {np.std(per_seed_lift[p], ddof=1) if len(used_seeds)>1 else 0.0:.4f}")

    save_kw = {'seeds': np.array(used_seeds), 'phases': np.array(PHASES),
               'n_rollouts': args.n_rollouts, 'months': args.months}
    for p in PHASES:
        save_kw[f'per_seed_lift_{p}'] = np.array(per_seed_lift[p])
        save_kw[f'per_seed_agent_{p}'] = np.array(per_seed_agent[p])
        save_kw[f'per_seed_base_{p}'] = np.array(per_seed_base[p])
        save_kw[f'per_seed_freq_agent_{p}'] = np.array(per_seed_freq_a[p])
        save_kw[f'per_seed_freq_base_{p}'] = np.array(per_seed_freq_b[p])
        save_kw[f'per_seed_dur_agent_{p}'] = np.array(per_seed_dur_a[p])
        save_kw[f'per_seed_dur_base_{p}'] = np.array(per_seed_dur_b[p])
    for p, (_, _, _, _, _, _, _, pval) in zip(PHASES, rows):
        save_kw[f'p_{p}'] = pval
    np.savez(output_dir / 'lift_ensemble.npz', **save_kw)

    # Tidy CSVs alongside the npz (survive lost logs; easy notebook loading).
    summary_rows = [{'phase': p, 'agent': am, 'agent_ci95': aci,
                     'baseline': bm, 'baseline_ci95': bci,
                     'lift': lm, 'lift_ci95': lci, 'p': pval}
                    for (p, am, aci, bm, bci, lm, lci, pval) in rows]
    save_csv(output_dir / 'lift_ensemble.csv', summary_rows)
    per_seed_rows = [{'phase': p, 'seed': int(s),
                      'agent': float(per_seed_agent[p][i]),
                      'baseline': float(per_seed_base[p][i]),
                      'lift': float(per_seed_lift[p][i])}
                     for p in PHASES for i, s in enumerate(used_seeds)]
    save_csv(output_dir / 'lift_ensemble_per_seed.csv', per_seed_rows)

    # ---- Lift decomposition: frequency vs duration (mechanism of the lift) ----
    decomp_rows = []
    for p in PHASES:
        fa_m, fa_c = _mean_ci_nan(per_seed_freq_a[p])
        fb_m, fb_c = _mean_ci_nan(per_seed_freq_b[p])
        da_m, da_c = _mean_ci_nan(per_seed_dur_a[p])
        db_m, db_c = _mean_ci_nan(per_seed_dur_b[p])
        decomp_rows.append({'phase': p,
                            'freq_agent_per100yr': fa_m, 'freq_agent_ci95': fa_c,
                            'freq_base_per100yr': fb_m, 'freq_base_ci95': fb_c,
                            'dur_agent_mo': da_m, 'dur_agent_ci95': da_c,
                            'dur_base_mo': db_m, 'dur_base_ci95': db_c})
    _print_decomp_table(decomp_rows)
    save_csv(output_dir / 'lift_decomposition.csv', decomp_rows)
    decomp_seed_rows = [{'phase': p, 'seed': int(s),
                         'freq_agent_per100yr': per_seed_freq_a[p][i],
                         'freq_base_per100yr': per_seed_freq_b[p][i],
                         'dur_agent_mo': per_seed_dur_a[p][i],
                         'dur_base_mo': per_seed_dur_b[p][i]}
                        for p in PHASES for i, s in enumerate(used_seeds)]
    save_csv(output_dir / 'lift_decomposition_per_seed.csv', decomp_seed_rows)
    _plot_decomp_bars(decomp_rows, output_dir / 'lift_decomposition_bars.png')
    _plot_decomp_shift(per_seed_freq_a, per_seed_dur_a, per_seed_freq_b,
                       per_seed_dur_b, output_dir / 'lift_decomposition_shift.png')
    return rows


def main():
    parser = argparse.ArgumentParser(description="Phase-resolved MYE lift analysis (ensemble)")
    parser.add_argument("--model", type=str, default="ensemble", help="Ensemble model prefix")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)),
                        help="Ensemble seeds")
    parser.add_argument("--n-rollouts", type=int, default=100,
                        help="Paired rollouts per seed")
    parser.add_argument("--months", type=int, default=1200, help="Months per rollout")
    parser.add_argument("--master-seed", type=int, default=42, help="Seed for rollout seeds")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    args = parser.parse_args()

    suppress_warnings()
    output_dir = Path("plots") / args.model / "lift"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_wandb:
        from config import WandbConfig
        wc = WandbConfig()
        wandb.init(project=wc.project, entity=wc.entity,
                   name=datetime.now().strftime(r"lift %H:%M %d-%m-%y"),
                   group="analysis", job_type="lift-analysis",
                   tags=["lift", "phase", "analysis"],
                   config={"n_rollouts": args.n_rollouts, "months": args.months})

    print("=" * 70)
    print("MULTI-YEAR ENSO LIFT ANALYSIS (phase-resolved)")
    print("=" * 70)
    start = time.time()

    rows = run_ensemble(args, output_dir)

    _log_lift_wandb(rows)
    if wandb.run is not None:
        wandb.finish()

    el = time.time() - start
    print(f"\n{'='*70}\nLIFT ANALYSIS COMPLETE — {int(el//60)}m {int(el%60)}s\n{'='*70}")


if __name__ == "__main__":
    main()
