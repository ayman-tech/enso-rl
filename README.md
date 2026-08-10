# ENSO-RL — Reinforcement Learning for Multi-Year ENSO Control

<p align="left">
  <img alt="Python" src="https://img.shields.io/badge/python-3.12%2B-blue">
  <img alt="RL" src="https://img.shields.io/badge/RL-PPO%20(Stable--Baselines3)-green">
  <img alt="Climate model" src="https://img.shields.io/badge/climate%20model-XRO-lightgrey">
  <img alt="Status" src="https://img.shields.io/badge/status-research-orange">
</p>

A reinforcement-learning framework that trains a **PPO agent to steer the XRO stochastic
climate model** toward **multi-year El Niño / La Niña events (MYE)**, then uses a suite of
**explainable-AI attributions** to identify *which* climate modes causally drive that
control. The agent discovers control strategies that are then interrogated with
counterfactual, Shapley, and interventional analyses — recovering known ENSO physics
(warm-water-volume recharge, inter-basin coupling) from a purely data-driven policy.

> **Scientific question.** Which climate-mode forcings, and in what combination, most
> increase the probability of *multi-year* ENSO events — and are those drivers robust
> across independent methods and random seeds?

---

## Table of contents
- [Overview](#overview)
- [Key features](#key-features)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Data](#data)
- [Quickstart](#quickstart)
- [Method summary](#method-summary)
- [Training](#training)
- [Evaluation & analysis pipeline](#evaluation--analysis-pipeline)
- [Notebooks](#notebooks)
- [Reproducibility](#reproducibility)
- [Configuration](#configuration)
- [References](#references)
- [Citation](#citation)
- [License & contact](#license--contact)

---

## Overview

ENSO (the El Niño–Southern Oscillation) is modelled here with **XRO** (Zhao et al. 2024,
*Nature*), a fitted stochastic dynamical model of 10 coupled climate-mode indices. We wrap
XRO as a **continuing-control Gymnasium environment** in which an agent applies bounded
monthly forcings to **9 controllable modes** (Niño3.4 is observed, not actioned) with the
objective of maximising the long-run fraction of months spent in a **multi-year ENSO
event** (a same-sign |Niño3.4| > 0.5 σ run lasting ≳ 1 year).

The pipeline has two halves:

1. **Control** — train a PPO ensemble to raise the multi-year-ENSO probability above a
   free-running (zero-action) baseline; the gap is the **lift**.
2. **Attribution** — explain the learned control with removal-based and causal methods to
   rank the physical drivers and test their robustness.

## Key features

- **Continuing-task formulation with partial-episode bootstrapping (PEB).** Episodic resets
  are treated as *truncations* (value-bootstrapped), not terminations — the correct
  treatment for a task with no terminal state (Pardo et al. 2018).
- **In-environment observation normalization.** The 10 physical modes are z-scored against
  the observed climatology so no single mode (e.g. warm-water volume) dominates the policy
  input; raw physical state is preserved for the reward and all analyses.
- **Stabilised training.** Return normalization (`VecNormalize`) + value-function clipping +
  entropy regularization + `target_kl` eliminate the value-loss blow-up and late-training
  entropy collapse.
- **Best-checkpointing on the true objective.** The saved model is the one with the best
  *evaluated multi-year lift*, not the last step — captured via an atomic, in-place save.
- **10-seed ensembles with five-axis seed control** for reproducibility and
  randomness-sensitivity studies.
- **Full XAI attribution suite** — counterfactual ablation, coalitional Shapley,
  interventional (do-operator) XRO, precursor composites, and policy-facing saliency
  (integrated gradients, gradient sensitivity, mutual information).
- **Reproducible `make`-driven workflow** with `quick` (fast) and `robust` (publication)
  presets for every stage.

---

## Repository structure

```
enso-rl/
├── config/                     # Dataclass configs (single source of truth)
│   ├── env_config.py           # XRO env: threshold, action scaling, reward weights
│   ├── train_config.py         # PPO hyperparameters + training duration
│   └── wandb_config.py         # Weights & Biases settings
├── envs/
│   └── xro_env.py              # XROMultiYearEnv (Gymnasium continuing-control env)
├── callbacks/
│   ├── training_history_callback.py   # per-seed metrics + best-model checkpointing
│   └── wandb_callback.py
├── utils/
│   ├── data_processing.py      # data loading + XRO parameter preparation
│   ├── physics.py              # one-step XRO transition (xro_step)
│   ├── evaluation.py           # rollouts, lift, trajectory simulation
│   ├── enso_classifier.py      # ENSO event / multi-year-event labelling
│   ├── seeding.py              # five-axis seed resolution + ensemble discovery
│   └── model_io.py, results_io.py, nb_helper.py
├── scripts/
│   ├── train.py                # train one PPO agent
│   ├── train_ensemble.py       # train an N-seed ensemble in parallel
│   ├── evaluate.py             # single-model evaluation / lift / trajectory
│   └── analysis/               # driver-attribution + XAI computation scripts
│       ├── inference.py                # paired agent-vs-baseline rollouts → lift/seasonality
│       ├── counterfactual_analysis.py  # zero-ablation driver importance (ensemble)
│       ├── shapley_analysis.py         # coalitional Shapley over control channels
│       ├── interventional_xro.py       # agent-free do-operator interventions
│       ├── precursor_composite.py      # lead-time driver composites
│       ├── integrated_gradients.py     # policy obs→action saliency
│       ├── gradient_sensitivity.py     # policy gradient sensitivity
│       └── mutual_information.py        # obs–action mutual information
├── notebooks/                  # offline figure generation (read the .npz outputs)
├── data/                       # XRO_indices_oras5.nc (ORAS5 indices, 1979–2022)
├── models/                     # trained model checkpoints (<name>_seed<N>.zip)
├── plots/                      # per-model analysis outputs (.npz, .csv, figures)
├── Architecture.md             # full hyperparameter + reward-function reference
├── Makefile                    # reproducible workflow (quick / robust presets)
└── pyproject.toml              # dependencies (managed with uv)
```

---

## Installation

Requires **Python ≥ 3.12**. Dependencies are managed with [`uv`](https://docs.astral.sh/uv/).

```bash
# From the repo root:
uv sync            # creates .venv and installs all dependencies from uv.lock
```

Key dependencies: `stable-baselines3` (PPO), `gymnasium`, `xro`, `xarray`, `numpy`, `scipy`,
`wandb`, `pandas`, `matplotlib`.

## Data

The environment is fit on **ORAS5 climate-mode indices (1979–2022)** from the XRO project.
Place the file at `data/XRO_indices_oras5.nc`:

```bash
mkdir -p data
wget -c -P data/ https://github.com/senclimate/XRO/raw/main/data/XRO_indices_oras5.nc
```

---

## Quickstart

The `Makefile` is the primary interface. `name=<prefix>` labels the run; models are saved
to `models/<prefix>_seed<N>.zip` and analysis outputs to `plots/<prefix>/`.

```bash
# 1. Train a 10-seed ensemble + paired inference (fast preset)
make train-quick   name=ensemble

# 2. Driver-attribution suite (counterfactual + Shapley + interventional)
make xai-quick     name=ensemble

# --- or the full publication pipeline (longer training, more samples) ---
make full-robust   name=ensemble
```

Then open the notebooks in `notebooks/` to render figures from the generated `.npz` files.

**Duration presets** (env steps; 1 step = 1 month): `train-quick` uses 240k, `train-robust`
uses 600k. Override per-invocation, e.g.
`make train-ensemble name=ensemble total_timesteps=1200000`.

---

## Method summary

Full details (every hyperparameter and every reward term) are in
**[`Architecture.md`](Architecture.md)**. In brief:

**Environment (`XROMultiYearEnv`).** Continuing-control task; each step advances XRO one
month under the agent's forcing. Observation = 10 climatology-z-scored modes + seasonal
month + current event-duration feature (12-D). Action = 9-D in `[-1, 1]`, scaled per-mode
per-month by the observed typical monthly change. Episodic resets (`max_episode_steps`)
re-anchor to a real observed initial condition and are handled as **truncations** (PEB).

**Reward** = multi-year-duration reward − over-persistence penalty − state-plausibility
(Mahalanobis) penalty − action-effort cost. Both penalties use **saturating (tanh) ramps**
so a single extreme step cannot destabilise the critic; the plausibility threshold is
anchored to the observed climatology envelope. See [`Architecture.md`](Architecture.md) §4.

**Algorithm.** PPO (Stable-Baselines3, `MlpPolicy`) with `gamma=0.95`, `n_steps=240`,
`n_epochs=10`, `clip_range_vf=0.2`, `ent_coef=0.003`, `target_kl=0.03`; the training env is
wrapped in `Monitor` + `VecNormalize(norm_reward=True)`. The reported metric — the
multi-year-ENSO **lift** over a zero-action baseline — drives **best-model checkpointing**.

---

## Training

```bash
# Single agent
uv run scripts/train.py --total-timesteps 240000 --name my-run

# Full ensemble (10 seeds, parallel)
uv run scripts/train_ensemble.py --prefix ensemble --n-seeds 10 --total-timesteps 600000 --no-wandb

# via Makefile
make train          name=my-run          # single, 240k default
make train-ensemble name=ensemble        # 10-seed ensemble
```

**`scripts/train.py` options**

| Flag | Description |
|---|---|
| `--total-timesteps N` | Total env steps for `model.learn()` (1 step = 1 month) |
| `--max-episode-steps N` | Episode length before a truncation/reset (default 1200 = 100 yr) |
| `--lr LR` | Learning-rate override |
| `--name NAME` | Run / model name |
| `--seed S` | Master seed (sets all five randomness axes) |
| `--seed-{weight,action,batch,init,physics} S` | Per-axis seed override (sensitivity study) |
| `--no-wandb` | Disable Weights & Biases logging |
| `--debug` | Verbose mode |

Entropy coefficient, target-KL, discount, and other PPO knobs live in
[`config/train_config.py`](config/train_config.py).

---

## Evaluation & analysis pipeline

All analysis scripts **auto-detect the trained ensemble** from `models/<name>_seed*.zip`
and write results to `plots/<name>/`. Each stage has a `quick` and a `robust` preset.

**Control performance**

```bash
make inference name=ensemble          # paired agent-vs-baseline rollouts → lift + seasonality
```

**Driver attribution** (the causal backbone)

```bash
make counterfactual name=ensemble     # zero-ablation importance of each control channel
make shapley        name=ensemble     # coalitional Shapley over the 9 control channels
make interventional name=ensemble     # agent-free do-operator interventions in XRO
make xai-quick      name=ensemble     # all three of the above
```

- **Counterfactual** and **Shapley** are complementary *removal-based* attributions
  (Covert, Lundberg & Lee 2021); their agreement across methods and across the 10 seeds is
  the robustness evidence.
- **Interventional XRO** is agent-free: it directly presses/brakes a mode in a free-running
  XRO and measures the causal ΔP(MYE) — the do-operator counterpart.

**Supporting analyses**

```bash
make precursor      name=ensemble     # lead-time driver composites (drivers vs spontaneous MYE)
make policy-facing  name=ensemble     # obs→action saliency (IG, gradient sensitivity, MI)
```

Single-model evaluation is available via `scripts/evaluate.py` (`--basic`, `--lift`,
`--trajectory`, `--intervention`, `--all`).

---

## Notebooks

Notebooks are **offline figure generators** — they read the `.npz` files produced by the
scripts above and never retrain.

| Notebook | Purpose |
|---|---|
| `notebooks/analysis.ipynb` | Behaviour, multi-year lift, and seasonality figures (from `inference.npz`) |
| `notebooks/analysis_xai.ipynb` | Driver attribution: counterfactual, Shapley, cross-method convergence, interventional, precursor, policy-facing saliency |
| `notebooks/train_analysis.ipynb` | Ensemble convergence: lift, raw reward, and PPO optimization diagnostics (median + IQR across seeds) |
| `notebooks/seed_sensitivity.ipynb` | Randomness-axis sensitivity study |

---

## Reproducibility

- **Five independent randomness axes** are seeded separately (policy-weight init, action
  sampling, mini-batch shuffle, env start state, XRO climate noise), enabling both fully
  reproducible runs and controlled one-axis-at-a-time sensitivity sweeps
  (`seed_axis_test.sh`, `notebooks/seed_sensitivity.ipynb`).
- **10-seed ensembles** are the unit of reporting; attribution significance is computed
  *across seeds* (per-seed means → confidence intervals / p-values). We recommend reporting
  IQM + bootstrapped CIs (Agarwal et al. 2021) for the final figures.
- The XRO climate-noise stream is isolated from PPO's global-RNG usage so evaluation cannot
  perturb training.

---

## Configuration

All configuration is declarative in `config/` (Python dataclasses):

| File | Controls |
|---|---|
| [`config/train_config.py`](config/train_config.py) | PPO hyperparameters, training duration, episode length, eval cadence |
| [`config/env_config.py`](config/env_config.py) | ENSO threshold, per-mode/per-month action scaling, reward weights & penalty caps |
| [`config/wandb_config.py`](config/wandb_config.py) | Weights & Biases project/entity/mode |

See [`Architecture.md`](Architecture.md) for the full annotated reference and the values
currently in use.

---

## References

- **XRO climate model** — Zhao, S. et al. (2024). *Explainable El Niño predictability from
  climate mode interactions.* **Nature**. (`senclimate/XRO`)
- **PPO** — Schulman, J. et al. (2017). *Proximal Policy Optimization Algorithms.*
  arXiv:1707.06347.
- **Stable-Baselines3** — Raffin, A. et al. (2021). *JMLR* 22(268).
- **Time limits / PEB** — Pardo, F. et al. (2018). *Time Limits in Reinforcement Learning.*
  ICML.
- **Removal-based explanations** — Covert, I., Lundberg, S. & Lee, S.-I. (2021). *Explaining
  by Removing: A Unified Framework for Model Explanation.* **JMLR**. / Lundberg & Lee (2017),
  *SHAP*, NeurIPS.
- **Evaluation rigor** — Henderson, P. et al. (2018), *Deep RL that Matters*; Agarwal, R.
  et al. (2021), *Deep RL at the Edge of the Statistical Precipice* (`rliable`).

## Citation

A manuscript describing this work is in preparation. Until then, please cite this
repository:

```bibtex
@software{enso_rl,
  title  = {ENSO-RL: Reinforcement Learning for Multi-Year ENSO Control},
  author = {Sayed, Ayman},
  year   = {2026},
  url    = {https://github.com/<your-org>/enso-rl}
}
```

## License & contact

No license file is currently included; please contact the author before reuse.
Questions and issues: **Ayman Sayed** · open a GitHub issue on this repository.
