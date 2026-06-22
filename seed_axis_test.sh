#!/usr/bin/env bash
#
# Randomness-sensitivity sweep (pin-four-vary-one), parallel.
#
# For each of the five randomness axes, the other four are pinned at $PIN while
# the target axis is swept over $SWEEP. The spread of the resulting mye_prob
# across each axis's runs is that axis's sensitivity ("common random numbers"
# attribution). Every axis is swept over the SAME value set so the per-axis
# variances are directly comparable.
#
#   axis     -> what its randomness controls
#   weight   -> policy weight initialization        (--seed-weight)
#   action   -> stochastic Gaussian exploration     (--seed-action)
#   batch    -> PPO mini-batch shuffle              (--seed-batch)
#   init     -> environment start state             (--seed-init)
#   physics  -> XRO climate noise                   (--seed-physics)
#
# Runs train in parallel ($JOBS at a time via xargs -P). Each run is ~1 core, so
# set $JOBS to the cores you have (defaults to SLURM_CPUS_PER_TASK or nproc) and
# export thread caps (OMP_NUM_THREADS=1 etc.) in your slurm script so the workers
# don't oversubscribe.
#
# Models:  baseline -> models/<PREFIX>_seed<PIN>.zip
#          sweep    -> models/<PREFIX>-<axis>_seed<w>-<a>-<b>-<i>-<p>.zip
# Per-run stdout goes to $LOGDIR/<axis>-<v>.log (baseline.log).
#
# Usage:
#   bash seed_axis_test.sh                          # defaults below
#   JOBS=8 SWEEP="1 2 3 4 5" bash seed_axis_test.sh
#   NO_WANDB=1 bash seed_axis_test.sh               # disable W&B logging

set -uo pipefail
cd "$(dirname "$0")"

# ---- configuration (override via environment) -------------------------------
EPOCHS="${EPOCHS:-1000}"          # training epochs per run
PIN="${PIN:-0}"                   # value the four fixed axes are held at
SWEEP="${SWEEP:-1 2 3 4 5}"       # values the target axis is swept over
PREFIX="${PREFIX:-sens}"          # model-name prefix
AXES="${AXES:-weight action batch init physics}"
JOBS="${JOBS:-${SLURM_CPUS_PER_TASK:-$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}}"
LOGDIR="${LOGDIR:-logs/sweep}"
EXTRA=""
[ "${NO_WANDB:-0}" = "1" ] && EXTRA="--no-wandb"

mkdir -p "$LOGDIR" models

# axis name -> train.py override flag
flag_for() {
  case "$1" in
    weight)  echo "--seed-weight" ;;
    action)  echo "--seed-action" ;;
    batch)   echo "--seed-batch" ;;
    init)    echo "--seed-init" ;;
    physics) echo "--seed-physics" ;;
    *) echo "unknown axis: $1" >&2; exit 1 ;;
  esac
}

# Emit one full command per line for every run in the sweep.
build_jobs() {
  # baseline: all axes pinned at $PIN -> name carries an explicit _seed<PIN> suffix
  echo "uv run scripts/train.py --epochs $EPOCHS --seed $PIN --name ${PREFIX}_seed${PIN} $EXTRA > $LOGDIR/baseline.log 2>&1"
  for axis in $AXES; do
    local flag; flag="$(flag_for "$axis")"
    for v in $SWEEP; do
      echo "uv run scripts/train.py --epochs $EPOCHS --seed $PIN $flag $v --name ${PREFIX}-${axis} $EXTRA > $LOGDIR/${axis}-${v}.log 2>&1"
    done
  done
}

n_sweep=$(echo $SWEEP | wc -w | tr -d ' ')
n_axes=$(echo $AXES | wc -w | tr -d ' ')
total=$(( 1 + n_axes * n_sweep ))

echo "=================================================================="
echo "RANDOMNESS-SENSITIVITY SWEEP (parallel)"
echo "  epochs/run = $EPOCHS | pin = $PIN | sweep = [$SWEEP] | prefix = $PREFIX"
echo "  axes       = $AXES"
echo "  jobs       = $JOBS parallel"
echo "  total runs = $total  (1 baseline + $n_axes axes x $n_sweep seeds)"
echo "  logs       = $LOGDIR/<axis>-<v>.log"
echo "=================================================================="

# Run up to $JOBS at a time. xargs returns nonzero if any job failed.
build_jobs | xargs -P "$JOBS" -I CMD bash -c CMD

echo; echo "=================================================================="
echo "SWEEP COMPLETE — models at models/${PREFIX}-<axis>_seed<w>-<a>-<b>-<i>-<p>.zip"
echo "Per axis, compare mye_prob across its $n_sweep runs to rank sensitivity."
echo "=================================================================="
