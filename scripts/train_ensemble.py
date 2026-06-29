"""
Train an ensemble of PPO agents with distinct seeds (paper point 1.3).

A single-seed result is not publishable: importance rankings and the mye_prob
lift must be reported as mean +/- CI across independently trained agents. The
cross-seed spread is itself a result — if several seeds reach high mye_prob via
different strategies, that is evidence of multiple pathways to multi-year ENSO.

Each agent is trained by invoking scripts/train.py with its own --seed and a
per-seed model name (models/<prefix>_seed<seed>.zip), so runs are fully
independent and reproducible.

Usage:
    uv run scripts/train_ensemble.py --n-seeds 10 --total-timesteps 240000 --no-wandb
    uv run scripts/train_ensemble.py --seeds 0 1 2 3 4 --total-timesteps 240000 --no-wandb
    uv run scripts/train_ensemble.py --n-seeds 10 --prefix ens --no-wandb
    uv run scripts/train_ensemble.py --n-seeds 10 --jobs 8 --no-wandb   # parallel
"""
import sys
import time
import argparse
import subprocess
from pathlib import Path
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))


def _build_cmd(seed, args, train_script):
    """Build the train.py command for one seed (identical to the serial invocation)."""
    name = f"{args.prefix}_seed{seed}"
    cmd = ["uv", "run", train_script,
           "--total-timesteps", str(args.total_timesteps),
           "--seed", str(seed),
           "--name", name]
    if args.lr is not None:
        cmd += ["--lr", str(args.lr)]
    if args.no_wandb:
        cmd += ["--no-wandb"]
    return name, cmd


def _run_one(seed, args, train_script, stream):
    """Run a single agent. Streams train.py output when stream=True (serial mode);
    otherwise captures it so parallel runs don't interleave on the terminal.

    The training itself is unchanged — this only launches the same subprocess, which
    inherits the parent environment (thread caps like OMP_NUM_THREADS are expected to
    be exported by the caller, e.g. the cluster slurm script).
    """
    name, cmd = _build_cmd(seed, args, train_script)
    t0 = time.time()
    if stream:
        print("  $ " + " ".join(cmd), flush=True)
        ret = subprocess.run(cmd, cwd=str(repo_root))
        out = ""
    else:
        ret = subprocess.run(cmd, cwd=str(repo_root),
                             capture_output=True, text=True)
        out = (ret.stdout or "") + (ret.stderr or "")
    return seed, name, ret.returncode == 0, time.time() - t0, out


def main():
    parser = argparse.ArgumentParser(description="Train an ensemble of seeded PPO agents")
    parser.add_argument("--n-seeds", type=int, default=10,
                        help="Number of seeds 0..n-1 (ignored if --seeds given)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Explicit list of seeds (overrides --n-seeds)")
    parser.add_argument("--total-timesteps", type=int, default=240_000,
                        help="Total training timesteps per agent")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate override")
    parser.add_argument("--prefix", type=str, default="ensemble_model",
                        help="Model name prefix; saved as models/<prefix>_seed<seed>")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel agents to train at once (default: cpu_count - 2; "
                             "use 1 for serial with live-streamed logs). Mirrors "
                             "shapley/counterfactual_analysis. Results are identical to "
                             "serial. Export thread caps (e.g. OMP_NUM_THREADS=1) in the "
                             "environment/slurm script if running many workers.")
    args = parser.parse_args()

    seeds = args.seeds if args.seeds is not None else list(range(args.n_seeds))
    train_script = str(repo_root / "scripts" / "train.py")
    n_workers = args.workers if args.workers else max(1, cpu_count() - 2)
    n_workers = min(n_workers, len(seeds))  # no more workers than tasks

    # Unlike the shapley/counterfactual pools (in-process NumPy → mp.Pool), each task
    # here just launches a train.py subprocess, so threads waiting on subprocess.run
    # are the right tool. The children inherit the parent environment, so per-process
    # thread caps (OMP_NUM_THREADS=1 etc.) are set via export in the launcher/slurm
    # script rather than here.

    print("=" * 70)
    print(f"ENSEMBLE TRAINING — {len(seeds)} agents, seeds = {seeds}")
    print(f"  total_timesteps/agent = {args.total_timesteps} | prefix = {args.prefix} | workers = {n_workers}")
    print("=" * 70)

    start = time.time()
    results = []

    if n_workers == 1:
        # Serial: stream each run's output live (unchanged behavior).
        for i, seed in enumerate(seeds):
            name = f"{args.prefix}_seed{seed}"
            print(f"\n[{i+1}/{len(seeds)}] seed={seed} -> models/{name}.zip")
            seed, name, ok, el, _ = _run_one(seed, args, train_script, stream=True)
            results.append((seed, name, ok))
            print(f"  {'[OK]' if ok else '[FAILED]'} seed={seed} ({el:.0f}s)", flush=True)
    else:
        # Parallel: capture each run's output and report on completion.
        print(f"\nLaunching up to {n_workers} agents in parallel (1 thread/process)...", flush=True)
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = {ex.submit(_run_one, seed, args, train_script, False): seed
                       for seed in seeds}
            for n_done, fut in enumerate(as_completed(futures), 1):
                seed, name, ok, el, out = fut.result()
                results.append((seed, name, ok))
                status = "[OK]" if ok else "[FAILED]"
                print(f"[{n_done}/{len(seeds)}] {status} seed={seed} -> "
                      f"models/{name}.zip ({el:.0f}s)", flush=True)
                if not ok:
                    tail = "\n".join(out.splitlines()[-20:])
                    print(f"  --- last output for seed={seed} ---\n{tail}\n", flush=True)
        results.sort(key=lambda r: r[0])  # restore seed order for the summary

    elapsed = time.time() - start
    h, rem = divmod(int(elapsed), 3600)
    m, s = divmod(rem, 60)
    print("\n" + "=" * 70)
    print(f"ENSEMBLE TRAINING COMPLETE — {h}h {m}m {s}s")
    n_ok = sum(1 for _, _, ok in results if ok)
    print(f"  {n_ok}/{len(seeds)} succeeded")
    for seed, name, ok in results:
        print(f"    seed={seed:<4} {'OK ' if ok else 'FAIL'}  models/{name}.zip")
    print("=" * 70)
    print(f"\nNext: analyze with\n"
          f"  uv run scripts/analysis/lift_analysis.py --ensemble --prefix {args.prefix} --seeds {' '.join(map(str, seeds))}")

    if n_ok < len(seeds):
        sys.exit(1)


if __name__ == "__main__":
    main()
