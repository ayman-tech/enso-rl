"""Paired post-onset forcing removal; computation only.

Run from repository root:
  uv run scripts/analysis/timed_ablation.py --model m1
Pilot:
  uv run scripts/analysis/timed_ablation.py --model m1 --seeds 0 --n-runs 1 --months 120 --output /tmp/timed-pilot

Onset t is a centered-index threshold crossing, confirmed at state t+1.
Branches start at that confirmation, before any persistence outcome is known.
Lag k means an action applied to state t+k, advancing it to t+k+1.
The endpoint is 13 consecutive same-phase centered months t..t+12.
All branches retain the policy's feedback on other channels, and resume full
control after the removal window. Effects concern this onset cohort and policy,
not natural ENSO or the effect of pre-onset forcing.
"""
import argparse
import copy
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DYNAMIC = ('state', 'state_unclipped', 'current_step', 'month_offset',
           'enso_history', 'consecutive_enso_months', 'enso_phase_sign')


def snapshot(env):
    result = {k: copy.deepcopy(getattr(env, k)) for k in DYNAMIC}
    result['noise'] = copy.deepcopy(env.rng_physics.bit_generator.state)
    return result


def restore(env, saved):
    for k in DYNAMIC:
        setattr(env, k, copy.deepcopy(saved[k]))
    env.rng_physics.bit_generator.state = copy.deepcopy(saved['noise'])


def persists(raw, sign, threshold):
    """raw contains t-1..t+13, yielding 13 complete centered means."""
    raw = np.asarray(raw, dtype=float)
    assert raw.shape == (15,)
    return bool(np.all(sign * (raw[:-2] + raw[1:-1] + raw[2:]) / 3 >= threshold))


def branch(env, model, saved, sign, channel=None, lag=1, width=3):
    restore(env, saved)
    raw = list(env.enso_history[-3:])
    obs = env._get_obs()
    for k in range(1, 13):
        action, _ = model.predict(obs, deterministic=True)
        action = action.copy()
        if channel is not None and lag <= k < lag + width:
            action[channel] = 0.0
        obs, _, _, _, _ = env.step(action)
        raw.append(float(env.state[0]))
    return persists(raw, sign, env.threshold), np.asarray(raw)


def worker(job):
    args, seed = job
    import torch
    from utils import suppress_warnings
    suppress_warnings()
    torch.set_num_threads(1)
    from utils.model_io import load_environment, resolve_env_config, envconfig_path
    name = f'{args.model}_seed{seed}'
    if not envconfig_path(name).exists():
        raise FileNotFoundError(f'Missing trained environment config for {name}')
    config = resolve_env_config(name)
    model, env, names = load_environment(name, config)
    env.max_steps = None
    channels = [list(names[1:]).index(m) for m in args.features]
    rng = np.random.default_rng(args.master_seed)
    rollout_seeds = rng.integers(0, 2**31-1, size=args.n_runs)
    records, controls, interventions = [], [], []
    control_traces, intervention_traces = [], []
    steps = 0
    begin = time.perf_counter()
    for ri, rs in enumerate(rollout_seeds):
        obs, _ = env.reset(seed=int(rs))
        previous = None
        for month in range(1, args.months + 1):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, _, _, _ = env.step(action)
            steps += 1
            if month < 2:
                continue
            oni = np.mean(env.enso_history[-3:], dtype=np.float64)
            phase = 1 if oni >= env.threshold else -1 if oni <= -env.threshold else 0
            onset = month - 1
            if previous is not None and phase and phase != previous and onset >= args.spinup:
                saved = snapshot(env)
                control, reference = branch(env, model, saved, phase)
                # A no-removal replay checks state and noise restoration for every event.
                sham, replay = branch(env, model, saved, phase)
                if sham != control or not np.array_equal(reference, replay):
                    raise RuntimeError('Paired replay failed')
                effects = np.empty((len(channels), len(args.lags)), dtype=bool)
                traces = np.empty((len(channels), len(args.lags), 15))
                for fi, channel in enumerate(channels):
                    for li, lag in enumerate(args.lags):
                        effects[fi, li], traces[fi, li] = branch(env, model, saved, phase, channel, lag, args.width)
                steps += 12 * (2 + effects.size)
                records.append((ri, int(rs), onset, phase))
                controls.append(control)
                interventions.append(effects)
                control_traces.append(reference)
                intervention_traces.append(traces)
                restore(env, saved)
                obs = env._get_obs()
            previous = phase
    controls = np.asarray(controls, dtype=bool)
    interventions = np.asarray(interventions, dtype=bool).reshape(-1, len(channels), len(args.lags))
    records = np.asarray(records, dtype=np.int64).reshape(-1, 4)
    delta = np.full((2, len(channels), len(args.lags)), np.nan)
    counts = np.zeros(2, dtype=int)
    probabilities = np.full(2, np.nan)
    for pi, phase in enumerate((1, -1)):
        selected = records[:, 3] == phase
        counts[pi] = selected.sum()
        if counts[pi]:
            probabilities[pi] = controls[selected].mean()
            delta[pi] = (interventions[selected].astype(float) - controls[selected, None, None]).mean(0)
    elapsed = time.perf_counter()-begin
    np.savez_compressed(Path(args.output)/f'seed{seed}.npz',
        records=records, record_columns=['rollout_index','rollout_seed','onset_state_index','phase_sign'],
        control_success=controls, intervention_success=interventions,
        control_nino34=np.asarray(control_traces).reshape(-1,15),
        intervention_nino34=np.asarray(intervention_traces).reshape(-1,len(channels),len(args.lags),15),
        trace_state_lags=np.arange(-1,14),
        per_seed_delta=delta, event_counts=counts, control_probability=probabilities,
        rollout_seeds=rollout_seeds, env_config=json.dumps(asdict(config)),
        features=args.features, lags=args.lags, width=args.width,
        elapsed_seconds=elapsed, simulated_steps=steps)
    print(f'seed {seed}: events EN/LN={counts.tolist()}, {steps} steps, {elapsed:.1f}s ({steps/max(elapsed,1e-9):.0f} steps/s)', flush=True)
    return delta, counts, probabilities, elapsed, steps


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', default='m1')
    parser.add_argument('--seeds', nargs='+', type=int, default=list(range(10)))
    parser.add_argument('--n-runs', type=int, default=30)
    parser.add_argument('--months', type=int, default=1200)
    parser.add_argument('--spinup', type=int, default=12)
    parser.add_argument('--features', nargs='+', default=['WWV','IOB','NPMM','SPMM'])
    parser.add_argument('--lags', nargs='+', type=int, default=[1,4,7,10])
    parser.add_argument('--width', type=int, default=3)
    parser.add_argument('--master-seed', type=int, default=20261022)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.n_runs<1 or args.months<=args.spinup+2 or args.spinup<2:
        parser.error('Require positive runs, spinup >= 2, months > spinup+2')
    if args.width<1 or any(k<1 or k+args.width>13 for k in args.lags):
        parser.error('Removal windows must fit action lags 1..12')
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.features))!=len(args.features) or args.lags!=sorted(set(args.lags)):
        parser.error('Seeds/features must be unique; lags unique and increasing')
    args.workers = min(len(args.seeds), max(1, cpu_count() - 2))
    print(f'Using {args.workers} workers for {len(args.seeds)} agents', flush=True)
    args.output = args.output or ROOT/'plots'/args.model/'timing_interventions'
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.glob('*.npz')):
        parser.error('Output already contains NPZ files; use a new --output directory')
    manifest = vars(args).copy();manifest['output']=str(args.output)
    manifest['git_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    manifest['script_source']=Path(__file__).read_text()
    (args.output/'run_config.json').write_text(json.dumps(manifest,indent=2))
    begin=time.perf_counter()
    jobs=[(args,seed) for seed in args.seeds]
    if args.workers==1:
        results=[worker(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers,len(jobs))) as pool:
            results=list(pool.map(worker,jobs))
    values=np.stack([r[0] for r in results])
    mean=values.mean(0)
    ci=student_t.ppf(.975,len(values)-1)*values.std(0,ddof=1)/np.sqrt(len(values)) if len(values)>1 else np.full_like(mean,np.nan)
    np.savez_compressed(args.output/'timing_interventions.npz',
        schema_version=1, features=args.features, phases=['el_nino','la_nina'],
        lags=args.lags, seeds=args.seeds, width=args.width,
        per_seed_delta=values, mean_delta=mean, ci95=ci,
        event_counts=np.stack([r[1] for r in results]), control_probability=np.stack([r[2] for r in results]),
        nino34_smooth_alignment='centered', nino34_smooth_months=3,
        outcome_label='Fraction of onset cohort persisting for 13 consecutive centered months',
        intervention_description=f'{args.width}-month forcing removal after onset confirmation; other channels retain policy feedback',
        lag_convention='Action at state onset+k advances state to onset+k+1; first intervention k>=1',
        cohort_definition='All centered threshold crossings after spinup; no future-duration selection',
        total_seconds=time.perf_counter()-begin)
    print(f'Saved {args.output}; wall time {time.perf_counter()-begin:.1f}s',flush=True)
    if not np.isfinite(values).all():
        print('WARNING: some agents have no events for a phase; increase collection length before plotting.')


if __name__=='__main__':
    main()
