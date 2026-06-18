"""
Multi-year-ENSO probability LIFT analysis, phase-resolved (paper points 1.1-1.3).

Reports the headline result of the paper: how much the RL agent raises the
probability of multi-year ENSO over a free-running (zero-action) baseline, with
confidence intervals, split into El Nino vs La Nina (the multi-year-La-Nina
asymmetry).

Two modes:
  * single model  : CIs computed across independent paired rollouts of one agent
                    (points 1.1 lift, 1.2 phase split).
  * --ensemble    : CIs computed across independently trained seeds; each seed
                    contributes its own mean lift (point 1.3). Also reports the
                    cross-seed spread, which evidences strategy degeneracy.

Each agent rollout is paired with a baseline rollout using the SAME seed, so the
lift is a clean within-seed difference (identical start state + noise sequence).

Usage:
    # Single model (1.1 + 1.2)
    uv run scripts/analysis/lift_analysis.py --model rl_model --n-rollouts 100 --months 1200

    # Ensemble (1.1 + 1.2 + 1.3)
    uv run scripts/analysis/lift_analysis.py --ensemble --prefix rl_model \
        --seeds 0 1 2 3 4 5 6 7 8 9 --n-rollouts 30 --months 1200
"""
import sys
import time
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import wandb
from pathlib import Path
from datetime import datetime
from scipy import stats as sp_stats

repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from utils import suppress_warnings
from utils.results_io import save_csv
from utils.evaluation import rollout_mye_phased
from config import EnvConfig
# NOTE: load_environment is imported lazily inside run_single/run_ensemble to
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

    Returns dict per phase: arrays of agent, baseline, and lift across rollouts.
    """
    rng = np.random.default_rng(master_seed)
    seeds = [int(rng.integers(0, 2**31)) for _ in range(n_rollouts)]

    agent = {p: np.zeros(n_rollouts) for p in PHASES}
    base = {p: np.zeros(n_rollouts) for p in PHASES}

    for i, seed in enumerate(seeds):
        a = rollout_mye_phased(env, agent=model, num_months=months, seed=seed)
        b = rollout_mye_phased(env, agent=None, num_months=months, seed=seed)
        for p in PHASES:
            agent[p][i] = a[p]
            base[p][i] = b[p]

    lift = {p: agent[p] - base[p] for p in PHASES}
    return {'agent': agent, 'baseline': base, 'lift': lift, 'seeds': seeds}


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


def _plot_lift(rows, out_path, title):
    labels = [PHASE_LABELS[p] for p, *_ in rows]
    lifts = [r[5] for r in rows]
    cis = [r[6] for r in rows]
    colors = ['#455A64', '#D32F2F', '#1976D2']  # total, el nino, la nina

    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(labels))
    ax.bar(x, lifts, yerr=cis, capsize=6, color=colors[:len(labels)],
           edgecolor='black', linewidth=1.2)
    for i, (lm, lci) in enumerate(zip(lifts, cis)):
        ax.text(i, lm + np.sign(lm or 1) * (lci + 0.005),
                f"{lm:+.3f}", ha='center',
                va='bottom' if lm >= 0 else 'top', fontsize=11, fontweight='bold')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('MYE probability lift (agent - baseline)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved {out_path}")


def _log_lift_wandb(rows, output_dir, img_name='lift_single.png'):
    """Log lift rows + plot into the CURRENTLY ACTIVE wandb run.

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
    img_path = output_dir / img_name
    if img_path.exists():
        log["lift_plot"] = wandb.Image(str(img_path))
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
    _plot_lift(rows, output_dir / 'lift_single.png', f"MYE Lift by Phase — {label}")
    save_kw = {'phases': np.array(PHASES), 'n_rollouts': n_rollouts, 'months': months}
    for p in PHASES:
        save_kw[f'agent_{p}'] = res['agent'][p]
        save_kw[f'baseline_{p}'] = res['baseline'][p]
        save_kw[f'lift_{p}'] = res['lift'][p]
    np.savez(output_dir / 'lift_single.npz', **save_kw)

    if wandb_log:
        _log_lift_wandb(rows, output_dir, 'lift_single.png')
    return rows


def run_single(args, output_dir):
    from scripts.evaluate import load_environment  # lazy: avoid circular import
    env_config = EnvConfig()
    model, env, _ = load_environment(args.model, env_config)
    # Standalone mode owns its own wandb run, so log via main() (wandb_log=False here).
    return run_lift_evaluation(model, env, n_rollouts=args.n_rollouts,
                               months=args.months, master_seed=args.master_seed,
                               output_dir=output_dir, label=args.model, wandb_log=False)


def run_ensemble(args, output_dir):
    from scripts.evaluate import load_environment  # lazy: avoid circular import
    env_config = EnvConfig()
    # Per-seed mean lift -> CIs computed ACROSS seeds (point 1.3)
    per_seed_lift = {p: [] for p in PHASES}
    per_seed_agent = {p: [] for p in PHASES}
    per_seed_base = {p: [] for p in PHASES}
    used_seeds = []

    for seed in args.seeds:
        name = f"{args.prefix}_seed{seed}"
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

    _print_table(f"MYE LIFT — ensemble '{args.prefix}' "
                 f"(N={len(used_seeds)} seeds x {args.n_rollouts} rollouts)", rows)

    # Cross-seed spread (strategy degeneracy signal)
    print(f"\n  Cross-seed spread (std of per-seed mean lift):")
    for p in PHASES:
        print(f"    {PHASE_LABELS[p]:<22}: std = {np.std(per_seed_lift[p], ddof=1) if len(used_seeds)>1 else 0.0:.4f}")

    _plot_lift(rows, output_dir / 'lift_ensemble.png',
               f"MYE Lift by Phase — ensemble (N={len(used_seeds)} seeds)")
    save_kw = {'seeds': np.array(used_seeds), 'phases': np.array(PHASES),
               'n_rollouts': args.n_rollouts, 'months': args.months}
    for p in PHASES:
        save_kw[f'per_seed_lift_{p}'] = np.array(per_seed_lift[p])
        save_kw[f'per_seed_agent_{p}'] = np.array(per_seed_agent[p])
        save_kw[f'per_seed_base_{p}'] = np.array(per_seed_base[p])
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
    return rows


def main():
    parser = argparse.ArgumentParser(description="Phase-resolved MYE lift analysis (1.1-1.3)")
    parser.add_argument("--model", type=str, default="rl_model", help="Model name (single mode)")
    parser.add_argument("--ensemble", action="store_true", help="Aggregate across seeds (1.3)")
    parser.add_argument("--prefix", type=str, default="rl_model", help="Ensemble model prefix")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)),
                        help="Ensemble seeds")
    parser.add_argument("--n-rollouts", type=int, default=100,
                        help="Paired rollouts per model")
    parser.add_argument("--months", type=int, default=1200, help="Months per rollout")
    parser.add_argument("--master-seed", type=int, default=42, help="Seed for rollout seeds")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    args = parser.parse_args()

    suppress_warnings()
    # Namespace by model name (single) or ensemble prefix: plots/<name>/lift
    name = args.prefix if args.ensemble else args.model
    output_dir = Path("plots") / name / "lift"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_wandb:
        from config import WandbConfig
        wc = WandbConfig()
        wandb.init(project=wc.project, entity=wc.entity,
                   name=datetime.now().strftime(r"lift %H:%M %d-%m-%y"),
                   group="analysis", job_type="lift-analysis",
                   tags=["lift", "phase", "analysis"],
                   config={"ensemble": args.ensemble, "n_rollouts": args.n_rollouts,
                           "months": args.months})

    print("=" * 70)
    print("MULTI-YEAR ENSO LIFT ANALYSIS (phase-resolved)")
    print("=" * 70)
    start = time.time()

    rows = run_ensemble(args, output_dir) if args.ensemble else run_single(args, output_dir)

    img = 'lift_ensemble.png' if args.ensemble else 'lift_single.png'
    _log_lift_wandb(rows, output_dir, img)
    if wandb.run is not None:
        wandb.finish()

    el = time.time() - start
    print(f"\n{'='*70}\nLIFT ANALYSIS COMPLETE — {int(el//60)}m {int(el%60)}s\n{'='*70}")


if __name__ == "__main__":
    main()
