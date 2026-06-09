.PHONY: train evaluate shapley counterfactual igi

name ?= model

train:
	uv run scripts/train.py --epochs 2000 --name $(name)
	uv run scripts/evaluate.py --model $(name) --trajectory
ensemble:
	uv run scripts/train_ensemble.py --n-seeds 10 --epochs 1000 --no-wandb

evaluate:
	uv run scripts/evaluate.py --model $(name) --all

shapley:
	uv run scripts/shapley_analysis.py --model $(model) --months 1200 --metric mye_prob --n-runs 30 --n-permutations 20

counterfactual:
	uv run scripts/analysis/counterfactual_analysis.py --model $(model) --n-runs 30 --months 1200