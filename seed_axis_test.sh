#!/usr/bin/env bash
#
# Randomness-sensitivity sweep (pin-four-vary-one).
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
# Each run saves models/<PREFIX>_seed<w>-<a>-<b>-<i>-<p>.zip and logs all five
# effective seeds. Total runs = 1 baseline + (#axes * #SWEEP).
#
# Usage:
#   bash seed_axis_test.sh                 # defaults below
#   EPOCHS=1000 SWEEP="1 2 3 4 5" bash seed_axis_test.sh
#   NO_WANDB=1 bash seed_axis_test.sh      # disable W&B logging

set -euo pipefail
cd "$(dirname "$0")"

# ---- configuration (override via environment) -------------------------------
EPOCHS="${EPOCHS:-1000}"          # training epochs per run
PIN="${PIN:-0}"                   # value the four fixed axes are held at
SWEEP="${SWEEP:-1 2 3 4 5}"       # values the target axis is swept over
PREFIX="${PREFIX:-sens}"          # model-name prefix
AXES="${AXES:-weight action batch init physics}"
EXTRA=""
[ "${NO_WANDB:-0}" = "1" ] && EXTRA="--no-wandb"

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

n_sweep=$(echo $SWEEP | wc -w | tr -d ' ')
n_axes=$(echo $AXES | wc -w | tr -d ' ')
total=$(( 1 + n_axes * n_sweep ))
echo "=================================================================="
echo "RANDOMNESS-SENSITIVITY SWEEP"
echo "  epochs/run = $EPOCHS | pin = $PIN | sweep = [$SWEEP]"
echo "  axes       = $AXES"
echo "  total runs = $total  (1 baseline + $n_axes axes x $n_sweep seeds)"
echo "=================================================================="

# ---- baseline: all five axes pinned at $PIN ---------------------------------
echo; echo "### BASELINE  --seed $PIN  (all axes = $PIN)"
uv run scripts/train.py --epochs "$EPOCHS" --seed "$PIN" --name "$PREFIX" $EXTRA

# ---- one sweep per axis -----------------------------------------------------
for axis in $AXES; do
  flag="$(flag_for "$axis")"
  echo; echo "######################## AXIS: $axis ($flag) ########################"
  for v in $SWEEP; do
    echo; echo "### axis=$axis  $flag=$v  (others pinned at $PIN)"
    uv run scripts/train.py --epochs "$EPOCHS" --seed "$PIN" "$flag" "$v" --name "$PREFIX-$axis" $EXTRA
  done
done

echo; echo "=================================================================="
echo "SWEEP COMPLETE — models saved as models/${PREFIX}_seed<w>-<a>-<b>-<i>-<p>.zip"
echo "Per axis, compare the mye_prob across its $n_sweep runs to rank sensitivity."
echo "=================================================================="
