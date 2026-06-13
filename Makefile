.PHONY: train evaluate shapley counterfactual igi ensemble shapley-ensemble counterfactual-ensemble shapley-ensemble-robust counterfactual-ensemble-robust interventional convergence

name ?= model

train:
	uv run scripts/train.py --epochs 2000 --name $(name)
	uv run scripts/evaluate.py --model $(name) --trajectory
ensemble:
	uv run scripts/train_ensemble.py --prefix $(name) --n-seeds 10 --epochs 100 --no-wandb
	uv run scripts/analysis/lift_analysis.py --ensemble --prefix $(name) --seeds 0 1 2 3 4 5 6 7 8 9 --no-wandb
	
evaluate:
	uv run scripts/evaluate.py --model $(name) --all

shapley:
	uv run scripts/analysis/shapley_analysis.py --model $(name) --months 1200 --metric mye_prob --n-runs 30 --n-permutations 20

counterfactual:
	uv run scripts/analysis/counterfactual_analysis.py --model $(name) --n-runs 30 --months 1200

# --- Quick variants (fast local iteration / sanity checks) ---
# Shared across methods: same 10 seeds + 1200mo (so rankings are comparable).
# Method-specific sampling (n-runs / n-permutations) tuned for speed here.
shapley-ensemble:
	uv run scripts/analysis/shapley_analysis.py --ensemble --prefix $(name) --seeds 0 1 2 3 4 5 6 7 8 9 --n-runs 6 --n-permutations 6 --months 1200 --no-wandb

counterfactual-ensemble:
	uv run scripts/analysis/counterfactual_analysis.py --ensemble --prefix $(name) --seeds 0 1 2 3 4 5 6 7 8 9 --n-runs 20 --months 1200 --no-wandb

# --- Robust variants (paper-grade; run on cluster) ---
# Same 10 seeds + 1200mo as the quick variants; sampling cranked to convergence.
shapley-ensemble-robust:
	uv run scripts/analysis/shapley_analysis.py --ensemble --prefix $(name) --seeds 0 1 2 3 4 5 6 7 8 9 --n-runs 30 --n-permutations 30 --months 1200 --no-wandb

counterfactual-ensemble-robust:
	uv run scripts/analysis/counterfactual_analysis.py --ensemble --prefix $(name) --seeds 0 1 2 3 4 5 6 7 8 9 --n-runs 100 --months 1200 --no-wandb

# --- Agent-free causal backbone + 3-method convergence figure ---
# interventional_xro is agent-free; --prefix is a label so its output files
# alongside the ensemble's so `convergence` can find all three npz inputs.
interventional:
	uv run scripts/analysis/interventional_xro.py --prefix $(name) --n-runs 30 --months 1200

convergence:
	uv run scripts/analysis/driver_convergence.py --prefix $(name)

traj-ensemble:
	uv run scripts/evaluate.py --model $(name)_seed0 --trajectory --no-wandb
	uv run scripts/evaluate.py --model $(name)_seed1 --trajectory --no-wandb
	uv run scripts/evaluate.py --model $(name)_seed2 --trajectory --no-wandb
	uv run scripts/evaluate.py --model $(name)_seed3 --trajectory --no-wandb
	uv run scripts/evaluate.py --model $(name)_seed4 --trajectory --no-wandb
	uv run scripts/evaluate.py --model $(name)_seed5 --trajectory --no-wandb
	uv run scripts/evaluate.py --model $(name)_seed6 --trajectory --no-wandb
	uv run scripts/evaluate.py --model $(name)_seed7 --trajectory --no-wandb
	uv run scripts/evaluate.py --model $(name)_seed8 --trajectory --no-wandb
	uv run scripts/evaluate.py --model $(name)_seed9 --trajectory --no-wandb
