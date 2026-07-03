# Main runs
.PHONY: train train-ensemble inference inference-robust shapley counterfactual shapley-robust counterfactual-robust
# other runs
.PHONY : precursor precursor-robust interventional interventional-robust
# Pipelines
.PHONY : xai-quick xai-robust train-quick train-robust full-quick full-robust 

# Archived
.PHONY : traj-ensemble policy-facing policy-facing-robust shapley-single counterfactual-single

name ?= model
# Training duration (env steps; 1 step = 1 month). Overridable per-invocation, and
# set per-pipeline by train-quick / train-robust below.
total_timesteps ?= 240000

train:
	uv run scripts/train.py --total-timesteps $(total_timesteps) --name $(name)
train-ensemble:
	uv run scripts/train_ensemble.py --prefix $(name) --n-seeds 10 --total-timesteps $(total_timesteps) --no-wandb

# --- Inference: paired rollouts → raw per-step npz (lift + seasonality) ---
inference:
	uv run scripts/analysis/inference.py --model $(name) --n-rollouts 30 --months 1200
inference-robust:
	uv run scripts/analysis/inference.py --model $(name) --n-rollouts 100 --months 1200

# =============== X-AI methods ================
shapley:
	uv run scripts/analysis/shapley_analysis.py --model $(name) --seeds 0 1 2 3 4 5 6 7 8 9 --n-runs 6 --n-permutations 6 --months 1200 --no-wandb
shapley-robust:
	uv run scripts/analysis/shapley_analysis.py --model $(name) --seeds 0 1 2 3 4 5 6 7 8 9 --n-runs 30 --n-permutations 30 --months 1200 --no-wandb

counterfactual:
	uv run scripts/analysis/counterfactual_analysis.py --model $(name) --seeds 0 1 2 3 4 5 6 7 8 9 --n-runs 20 --months 1200 --no-wandb
counterfactual-robust:
	uv run scripts/analysis/counterfactual_analysis.py --model $(name) --seeds 0 1 2 3 4 5 6 7 8 9 --n-runs 100 --months 1200 --no-wandb

# --- Agent-free causal backbone + 3-method convergence figure ---
# interventional_xro is agent-free; --model is a label so its output files alongside the ensemble's so `convergence` can find all three npz inputs.
interventional:
	uv run scripts/analysis/interventional_xro.py --model $(name) --n-runs 30 --months 1200
interventional-robust:
	uv run scripts/analysis/interventional_xro.py --model $(name) --n-runs 50 --months 1200 --mode press --direction both


# ================== Other Analysis ===================
# precursor_composite is agent-free (no model); validates drivers vs spontaneous MYE (2.1).
precursor:
	uv run scripts/analysis/precursor_composite.py --model $(name) --n-runs 30 --months 1200 --lead 24
precursor-robust:
	uv run scripts/analysis/precursor_composite.py --model $(name) --n-runs 100 --months 2400 --lead 24

# Aggregate result pipelines (run on an already-trained ensemble: name=<prefix>)

xai-quick:
	$(MAKE) counterfactual name=$(name)
	$(MAKE) shapley name=$(name)
	$(MAKE) interventional name=$(name)
xai-robust:
	$(MAKE) counterfactual-robust name=$(name)
	$(MAKE) shapley-robust name=$(name)
	$(MAKE) interventional-robust name=$(name)

# quick = shorter training for fast pipeline validation/iteration + light inference.
# robust = longer training for convergence (publication model) + heavy inference.
# 240k was undertrained (policy_std still falling, KL/clip rising), so robust >> quick.
train-quick:
	$(MAKE) train-ensemble name=$(name) total_timesteps=240000
	$(MAKE) inference name=$(name)

train-robust:
	$(MAKE) train-ensemble name=$(name) total_timesteps=600000
	$(MAKE) inference-robust name=$(name)

full-quick:
	$(MAKE) train-quick name=$(name)
	$(MAKE) xai-quick name=$(name)

full-robust:
	$(MAKE) train-robust name=$(name)
	$(MAKE) xai-robust name=$(name)


# ============================== ARCHIVED ================================
# Policy-facing XAI (MI / gradient / IG): single-model scripts, run on a representative
# seed. These are SUPPORTING evidence (what the policy attends to), not causal.
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
policy-facing:
	uv run scripts/analysis/mutual_information.py   --model $(name)_seed0 --months 1200
	uv run scripts/analysis/gradient_sensitivity.py --model $(name)_seed0 --months 600 --n-runs 6
	uv run scripts/analysis/integrated_gradients.py --model $(name)_seed0 --months 600 --n-runs 6 --n-samples 100

policy-facing-robust:
	uv run scripts/analysis/mutual_information.py   --model $(name)_seed0 --months 6000
	uv run scripts/analysis/gradient_sensitivity.py --model $(name)_seed0 --months 1200 --n-runs 30
	uv run scripts/analysis/integrated_gradients.py --model $(name)_seed0 --months 1200 --n-runs 30 --n-samples 200

# single model analysis
shapley-single:
	uv run scripts/analysis/shapley_analysis.py --model $(name) --months 1200 --metric mye_prob --n-runs 30 --n-permutations 20
counterfactual-single:
	uv run scripts/analysis/counterfactual_analysis.py --model $(name) --n-runs 30 --months 1200
