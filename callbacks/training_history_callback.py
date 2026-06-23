"""
Training-history persistence callback (single writer for the convergence figures).

During training PPO only exposes the shaped reward, while the quantity we report
is the multi-year-ENSO probability (`mye_prob`). This callback owns ALL on-disk
training history so that ensemble convergence/reproducibility figures can be made
offline (W&B cannot overlay many seeds as one median + band). It records two
streams, each on its natural cadence, into a single per-seed file
`plots/<model_name>/training.npz` (+ a tidy `training.csv` sidecar):

  * dense optimization diagnostics, one row per PPO update (`_on_rollout_end`):
    smoothed episode reward/length plus the SB3 logger metrics (explained
    variance, approx-KL, clip fraction, entropy/value/policy losses, policy std);
  * sparse `mye_prob` evaluation, every `eval_freq` timesteps (`_on_step`): the
    current policy rolled out on a dedicated eval env, a zero-action baseline, and
    the lift over it.

The offline aggregator is `notebooks/train_analysis.ipynb`, which globs
`plots/<prefix>_seed*/training.npz` and plots median + IQR bands across seeds.
"""
import sys
import time
from pathlib import Path

import numpy as np
import wandb
from stable_baselines3.common.callbacks import BaseCallback

from utils.evaluation import evaluate_agent
from utils.results_io import save_csv


# SB3-logger keys captured once per update, paired with the short name persisted to
# disk. Missing on the very first `_on_rollout_end` (the update has not run yet) —
# stored as NaN and dropped by the notebook.
_DIAG_KEYS = {
    "train/explained_variance": "explained_variance",
    "train/approx_kl": "approx_kl",
    "train/clip_fraction": "clip_fraction",
    "train/entropy_loss": "entropy_loss",
    "train/value_loss": "value_loss",
    "train/policy_gradient_loss": "policy_gradient_loss",
    "train/std": "policy_std",  # mean action std — the exploration-collapse signal
}


class TrainingHistoryCallback(BaseCallback):
    """Record per-seed training history to `plots/<model_name>/training.{npz,csv}`.

    Args:
        eval_env: A SEPARATE environment instance (not the training env).
            `evaluate_agent` calls `reset()`, which would corrupt the in-progress
            PPO rollout if the training env were reused.
        model_name (str): On-disk run identifier (the train.py `save_name`); output
            goes to `plots/<model_name>/`.
        eval_freq (int): Evaluate `mye_prob` every this many timesteps.
        eval_steps (int): Months per evaluation rollout. Longer = lower-variance
            `mye_prob`, but multi-year events are rare so keep it reasonably long.
        n_episodes (int): Rollouts averaged per evaluation (each a fresh reset
            seed) for a more stable estimate.
        log_baseline (bool): Also evaluate the zero-action baseline and log the
            lift (agent - baseline). The baseline is policy-independent, so it is
            computed once and cached.
        verbose (int): Verbosity.
    """

    def __init__(self, eval_env, model_name, eval_freq=24000, eval_steps=1200,
                 n_episodes=3, log_baseline=True, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.eval_steps = eval_steps
        self.n_episodes = n_episodes
        self.log_baseline = log_baseline
        self._next_eval = eval_freq
        self._baseline_mye = None  # cached: baseline is policy-independent

        self.out_dir = Path("plots") / model_name
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self._train_rows = []  # one dict per PPO update
        self._eval_rows = []   # one dict per evaluation

        # Use the interpreter's original stdout: the training loop redirects
        # sys.stdout to a buffer around model.learn() (and constructs this callback
        # after that redirect), so sys.stdout here would be the buffer.
        # sys.__stdout__ is unaffected by reassignment and reaches the terminal.
        self._stdout = sys.__stdout__

    # ------------------------------------------------------------------ eval ---
    def _mean_mye(self, agent):
        """Average mye_prob over n_episodes independent rollouts.

        evaluate_agent() calls eval_env.reset() with no seed each time, so every
        rollout draws a fresh start state and noise sequence — the runs are
        independent and averaging them lowers the variance of the estimate.
        """
        vals = [
            evaluate_agent(self.eval_env, agent=agent,
                           continuous_steps=self.eval_steps)
            for _ in range(self.n_episodes)
        ]
        return float(np.mean(vals)), float(np.std(vals))

    def _run_eval(self):
        t0 = time.time()
        mye_mean, mye_std = self._mean_mye(self.model)

        row = {
            "timestep": int(self.num_timesteps),
            "mye_prob": mye_mean,
            "mye_prob_std": mye_std,
        }
        log_dict = {
            "eval/mye_prob": mye_mean,
            "eval/mye_prob_std": mye_std,
            "eval/timesteps": self.num_timesteps,
        }

        if self.log_baseline:
            if self._baseline_mye is None:
                self._baseline_mye, _ = self._mean_mye(agent=None)
            row["mye_baseline"] = self._baseline_mye
            row["mye_lift"] = mye_mean - self._baseline_mye
            log_dict["eval/mye_prob_baseline"] = self._baseline_mye
            log_dict["eval/mye_lift"] = row["mye_lift"]

        self._eval_rows.append(row)

        if wandb.run is not None:
            wandb.log(log_dict)

        # Persist on every eval: cheap (~20x/run) and crash-safe on the cluster.
        self._persist()

        if self.verbose > 0:
            msg = (f"[TrainHistory] t={self.num_timesteps:>8,} | "
                   f"mye_prob={mye_mean:.3f}±{mye_std:.3f}")
            if self.log_baseline:
                msg += (f" | baseline={self._baseline_mye:.3f} | "
                        f"lift={mye_mean - self._baseline_mye:+.3f}")
            msg += f" | {time.time() - t0:.1f}s"
            print(msg, file=self._stdout, flush=True)

    # -------------------------------------------------------------- diagnostics ---
    def _on_rollout_end(self) -> None:
        """One diagnostics row per PPO update (fires after each rollout collection).

        The primary reward signal is the mean step reward of the just-collected
        rollout buffer: this env trains continuously (`episode_length=None`, no
        resets), so `ep_info_buffer` stays empty and episode reward is unavailable.
        `ep_rew_mean` is still recorded best-effort (populated only if resets ever
        happen). Optimization metrics come from the SB3 logger and default to NaN
        until the first `train()` has populated them.
        """
        row = {"timestep": int(self.num_timesteps)}

        rb = getattr(self.model, "rollout_buffer", None)
        row["mean_reward"] = (float(np.mean(rb.rewards))
                              if rb is not None and rb.rewards is not None
                              else np.nan)

        ep_buf = getattr(self.model, "ep_info_buffer", None)
        if ep_buf:
            row["ep_rew_mean"] = float(np.mean([ep["r"] for ep in ep_buf]))
            row["ep_len_mean"] = float(np.mean([ep["l"] for ep in ep_buf]))
        else:
            row["ep_rew_mean"] = np.nan
            row["ep_len_mean"] = np.nan

        name_to_value = getattr(self.model.logger, "name_to_value", {}) or {}
        for sb3_key, short in _DIAG_KEYS.items():
            # Coerce with float() rather than isinstance: SB3 records some metrics as
            # numpy scalars (e.g. approx_kl is float32 via .cpu().numpy()), which are
            # not Python floats. float() handles numpy scalars/0-d arrays; missing
            # keys (first update) raise and fall back to NaN.
            try:
                row[short] = float(name_to_value[sb3_key])
            except (KeyError, TypeError, ValueError):
                row[short] = np.nan

        self._train_rows.append(row)

    # ---------------------------------------------------------------- plumbing ---
    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_eval:
            self._run_eval()
            self._next_eval += self.eval_freq
        return True

    def _on_training_end(self) -> None:
        # Final evaluation so the end-of-training point is always logged; _run_eval
        # persists, so disk reflects the complete run.
        self._run_eval()

    # ----------------------------------------------------------------- persist ---
    def _persist(self):
        """Write training.npz (arrays) + training.csv (tidy long). Full rewrite each
        time — the history is small (hundreds of update rows, ~20 eval rows)."""
        npz = {"model_name": self.model_name,
               "total_timesteps": int(self.num_timesteps)}

        # Dense train stream: one array per metric, aligned by train_timesteps.
        train_cols = ["mean_reward", "ep_rew_mean", "ep_len_mean", *(_DIAG_KEYS.values())]
        npz["train_timesteps"] = np.array(
            [r["timestep"] for r in self._train_rows], dtype=float)
        for col in train_cols:
            npz[col] = np.array([r.get(col, np.nan) for r in self._train_rows],
                                dtype=float)

        # Sparse eval stream.
        eval_cols = ["mye_prob", "mye_prob_std", "mye_baseline", "mye_lift"]
        npz["eval_timesteps"] = np.array(
            [r["timestep"] for r in self._eval_rows], dtype=float)
        for col in eval_cols:
            npz["eval_" + col] = np.array(
                [r.get(col, np.nan) for r in self._eval_rows], dtype=float)

        np.savez(self.out_dir / "training.npz", **npz)

        # Tidy/long CSV sidecar: one (split, timestep, metric, value) row each.
        rows = []
        for r in self._train_rows:
            for col in train_cols:
                rows.append({"split": "train", "timestep": r["timestep"],
                             "metric": col, "value": r.get(col, np.nan)})
        for r in self._eval_rows:
            for col in eval_cols:
                rows.append({"split": "eval", "timestep": r["timestep"],
                             "metric": col, "value": r.get(col, np.nan)})
        save_csv(self.out_dir / "training.csv", rows,
                 fieldnames=["split", "timestep", "metric", "value"])
