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
    uv run scripts/train_ensemble.py --n-seeds 10 --epochs 1000 --no-wandb
    uv run scripts/train_ensemble.py --seeds 0 1 2 3 4 --epochs 1000 --no-wandb
    uv run scripts/train_ensemble.py --n-seeds 10 --prefix ens --no-wandb
"""
import sys
import time
import argparse
import subprocess
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))


def main():
    parser = argparse.ArgumentParser(description="Train an ensemble of seeded PPO agents")
    parser.add_argument("--n-seeds", type=int, default=10,
                        help="Number of seeds 0..n-1 (ignored if --seeds given)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Explicit list of seeds (overrides --n-seeds)")
    parser.add_argument("--epochs", type=int, default=1000, help="Training epochs per agent")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate override")
    parser.add_argument("--prefix", type=str, default="ensemble_model",
                        help="Model name prefix; saved as models/<prefix>_seed<seed>")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    args = parser.parse_args()

    seeds = args.seeds if args.seeds is not None else list(range(args.n_seeds))
    train_script = str(repo_root / "scripts" / "train.py")

    print("=" * 70)
    print(f"ENSEMBLE TRAINING — {len(seeds)} agents, seeds = {seeds}")
    print(f"  epochs/agent = {args.epochs} | prefix = {args.prefix}")
    print("=" * 70)

    start = time.time()
    results = []
    for i, seed in enumerate(seeds):
        name = f"{args.prefix}_seed{seed}"
        cmd = ["uv", "run", train_script,
               "--epochs", str(args.epochs),
               "--seed", str(seed),
               "--name", name]
        if args.lr is not None:
            cmd += ["--lr", str(args.lr)]
        if args.no_wandb:
            cmd += ["--no-wandb"]

        print(f"\n[{i+1}/{len(seeds)}] seed={seed} -> models/{name}.zip")
        print("  $ " + " ".join(cmd), flush=True)
        ret = subprocess.run(cmd, cwd=str(repo_root))
        ok = ret.returncode == 0
        results.append((seed, name, ok))
        print(f"  {'[OK]' if ok else '[FAILED]'} seed={seed}", flush=True)

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
