"""
Seasonality of control (paper point 2.8) — when does the agent force, by calendar month?

The agent's scaled forcing is aggregated by calendar month x mode to expose *when* in
the seasonal cycle the controller acts, and tied to the spring predictability barrier
(boreal spring, Mar-May). Cheap because the scaled actions are already produced by
simulate_trajectory; this script only rolls out the trained agent(s) and bins them.

Calendar alignment (verified): the XRO emulator anchors its seasonal cycle to
January = phase 0 (fit_ds 'cycle' index 0 ~ mid-Jan; training starts 1979-01), and
xro_step rolls the operator by step % 12. So in a rollout, step % 12 maps directly to
calendar month (0 = Jan, 2-4 = Mar-May = spring barrier).

Caveat: env.reset() samples a random real start month but resets the seasonal clock to
phase 0, so the initial state's true month is discarded — a one-time spin-up seasonal
mismatch. We drop the first `--spinup` months of each rollout before aggregating.

Usage:
    uv run scripts/analysis/seasonality_of_control.py --prefix e100 --seeds 0 1 2 3 4
    uv run scripts/analysis/seasonality_of_control.py --model rl_model --months 6000
"""
import sys
import time
import argparse
import numpy as np
from pathlib import Path

repo_root = Path(__file__).parent.parent.parent
analysis_dir = Path(__file__).parent
sys.path.insert(0, str(analysis_dir))
sys.path.insert(0, str(repo_root))

from counterfactual_analysis import load_environment  # noqa: E402
from utils.evaluation import simulate_trajectory  # noqa: E402
from utils import suppress_warnings  # noqa: E402
from utils.results_io import save_csv  # noqa: E402
from config import EnvConfig  # noqa: E402

MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
SPRING_BARRIER = [2, 3, 4]  # Mar, Apr, May


def collect_seasonal_forcing(model, env, num_months, spinup, seed):
    """Roll out the agent and bin |scaled action| by calendar month x mode.

    Returns:
        abs_by_month:    [12, n_modes] mean |scaled action| per calendar month.
        signed_by_month: [12, n_modes] mean signed scaled action per calendar month.
        counts:          [12] number of contributing months per calendar slot.
    """
    sim = simulate_trajectory(env, agent=model, num_months=num_months, seed=seed)
    actions = sim['actions_traj']                  # [num_months, n_modes] (scaled)
    n = actions.shape[0]
    month_idx = np.arange(n) % 12                  # step 0 = January
    keep = np.arange(n) >= spinup                  # drop spin-up transient
    actions, month_idx = actions[keep], month_idx[keep]

    n_modes = actions.shape[1]
    abs_by_month = np.zeros((12, n_modes))
    signed_by_month = np.zeros((12, n_modes))
    counts = np.zeros(12, dtype=int)
    for m in range(12):
        sel = month_idx == m
        counts[m] = int(sel.sum())
        if counts[m]:
            abs_by_month[m] = np.abs(actions[sel]).mean(axis=0)
            signed_by_month[m] = actions[sel].mean(axis=0)
    return abs_by_month, signed_by_month, counts


def _resolve_models(args, env_config):
    """Return list of (label, model, env, var_names) for ensemble seeds."""
    names = [f"{args.model}_seed{s}" for s in args.seeds]
    loaded = []
    for name in names:
        path = Path(f"models/{name}.zip")
        if not path.exists():
            print(f"  [skip] models/{name}.zip not found")
            continue
        model, env, var_names = load_environment(name, env_config)
        loaded.append((name, model, env, var_names))
    return loaded


def _print_table(total_by_month, var_names, peak_modes):
    drivers = var_names[1:]
    print(f"\n{'='*70}")
    print("SEASONALITY OF CONTROL — total |scaled forcing| by calendar month")
    print('='*70)
    print(f"  {'Month':<6} {'TotalForcing':>14} {'Spring?':>9}")
    print('  ' + '-'*32)
    for m in range(12):
        flag = 'spring' if m in SPRING_BARRIER else ''
        print(f"  {MONTHS[m]:<6} {total_by_month[m]:>14.4f} {flag:>9}")
    spring = total_by_month[SPRING_BARRIER].sum()
    frac = spring / total_by_month.sum() if total_by_month.sum() else 0.0
    print('  ' + '-'*32)
    print(f"  Spring (Mar-May) share of annual forcing: {frac*100:.1f}% "
          f"(even split = 25.0%)")
    print(f"  Peak forcing month: {MONTHS[int(np.argmax(total_by_month))]}")
    print(f"  Most-forced modes: {', '.join(peak_modes)}")


def main():
    parser = argparse.ArgumentParser(description="Seasonality of control (calendar month x mode)")
    parser.add_argument("--model", type=str, default="ensemble",
                        help="Ensemble model prefix (models/{model}_seed{s}.zip)")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)),
                        help="Ensemble seeds to pool over")
    parser.add_argument("--months", type=int, default=6000, help="Months per rollout")
    parser.add_argument("--spinup", type=int, default=12,
                        help="Months to discard at start (seasonal spin-up)")
    parser.add_argument("--master-seed", type=int, default=42)
    args = parser.parse_args()

    suppress_warnings()
    env_config = EnvConfig()
    output_dir = Path("plots") / args.model / "seasonality"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SEASONALITY OF CONTROL")
    print("=" * 70)
    start = time.time()

    loaded = _resolve_models(args, env_config)
    if not loaded:
        raise FileNotFoundError("No models found. Check --model or --prefix/--seeds.")

    # Pool the per-month means across models (each model weighted equally).
    abs_stack, signed_stack = [], []
    var_names = None
    rng = np.random.default_rng(args.master_seed)
    for name, model, env, vnames in loaded:
        var_names = vnames
        seed = int(rng.integers(0, 2**31))
        abs_m, signed_m, counts = collect_seasonal_forcing(
            model, env, args.months, args.spinup, seed)
        abs_stack.append(abs_m)
        signed_stack.append(signed_m)
        print(f"  [ok] {name}: {int(counts.sum())} months binned "
              f"(>= {counts.min()}/month)")

    abs_by_month = np.mean(abs_stack, axis=0)          # [12, n_modes]
    signed_by_month = np.mean(signed_stack, axis=0)
    abs_sem = (np.std(abs_stack, axis=0, ddof=1) / np.sqrt(len(abs_stack))
               if len(abs_stack) > 1 else np.zeros_like(abs_by_month))
    total_by_month = abs_by_month.sum(axis=1)          # [12]

    # Modes that receive the most forcing overall (for the table)
    drivers = var_names[1:]
    mode_totals = abs_by_month.sum(axis=0)
    peak_modes = [drivers[i] for i in np.argsort(mode_totals)[::-1][:3]]

    _print_table(total_by_month, var_names, peak_modes)

    np.savez(
        output_dir / 'seasonality_of_control.npz',
        months=np.array(MONTHS),
        var_names=np.array(var_names),
        abs_by_month=abs_by_month,        # [12, n_modes] mean |scaled action|
        abs_sem=abs_sem,                  # across-model SEM
        signed_by_month=signed_by_month,  # [12, n_modes] mean signed action
        total_by_month=total_by_month,    # [12]
        spring_barrier=np.array(SPRING_BARRIER),
        models=np.array([n for n, *_ in loaded]),
        months_per_rollout=args.months, spinup=args.spinup,
    )
    print(f"  Saved {output_dir / 'seasonality_of_control.npz'}")

    # Tidy CSVs alongside the npz: long month x mode forcing + monthly totals.
    drivers = var_names[1:]
    long_rows = [{'month': MONTHS[m], 'month_idx': m, 'mode': drivers[j],
                  'mean_abs_forcing': float(abs_by_month[m, j]),
                  'mean_signed_forcing': float(signed_by_month[m, j])}
                 for m in range(12) for j in range(len(drivers))]
    save_csv(output_dir / 'seasonality_of_control.csv', long_rows)
    total_rows = [{'month': MONTHS[m], 'month_idx': m,
                   'total_abs_forcing': float(total_by_month[m]),
                   'spring_barrier': bool(m in SPRING_BARRIER)} for m in range(12)]
    save_csv(output_dir / 'seasonality_of_control_monthly_total.csv', total_rows)

    el = time.time() - start
    print(f"\n{'='*70}\nSEASONALITY OF CONTROL COMPLETE — {int(el//60)}m {int(el%60)}s\n{'='*70}")


if __name__ == "__main__":
    main()
