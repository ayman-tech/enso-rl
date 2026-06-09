"""
Periodic multi-year-ENSO (MYE) probability evaluation callback.

During training, PPO only exposes the shaped reward. The quantity we actually
care about and report is the multi-year-ENSO probability (`mye_prob`). This
callback periodically rolls out the current policy on a dedicated evaluation
environment and logs `mye_prob` (plus a zero-action baseline and the lift over
it) so convergence can be judged on the reported metric, not just on reward.
"""
import sys
import time
import numpy as np
import wandb
from stable_baselines3.common.callbacks import BaseCallback

from utils.evaluation import evaluate_agent


class MYEEvalCallback(BaseCallback):
    """Evaluate `mye_prob` every `eval_freq` timesteps.

    Args:
        eval_env: A SEPARATE environment instance (not the training env).
            `evaluate_agent` calls `reset()`, which would corrupt the in-progress
            PPO rollout if the training env were reused.
        eval_freq (int): Evaluate every this many timesteps.
        eval_steps (int): Months per evaluation rollout. Longer = lower-variance
            `mye_prob`, but multi-year events are rare so keep it reasonably long.
        n_episodes (int): Number of rollouts to average per evaluation (each with
            a different reset seed) for a more stable estimate.
        log_baseline (bool): Also evaluate the zero-action baseline and log the
            lift (agent - baseline). The baseline is independent of the policy,
            so it is computed once and cached.
        verbose (int): Verbosity.
    """

    def __init__(self, eval_env, eval_freq=24000, eval_steps=1200,
                 n_episodes=3, log_baseline=True, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.eval_steps = eval_steps
        self.n_episodes = n_episodes
        self.log_baseline = log_baseline
        self._next_eval = eval_freq
        self._baseline_mye = None  # cached: baseline is policy-independent
        # Use the interpreter's original stdout: the training loop redirects
        # sys.stdout to a buffer around model.learn() (and constructs this
        # callback after that redirect), so sys.stdout here would be the buffer.
        # sys.__stdout__ is unaffected by reassignment and reaches the terminal.
        self._stdout = sys.__stdout__

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

        log_dict = {
            "eval/mye_prob": mye_mean,
            "eval/mye_prob_std": mye_std,
            "eval/timesteps": self.num_timesteps,
        }

        if self.log_baseline:
            if self._baseline_mye is None:
                self._baseline_mye, _ = self._mean_mye(agent=None)
            log_dict["eval/mye_prob_baseline"] = self._baseline_mye
            log_dict["eval/mye_lift"] = mye_mean - self._baseline_mye

        if wandb.run is not None:
            wandb.log(log_dict)

        if self.verbose > 0:
            msg = (f"[MYEEval] t={self.num_timesteps:>8,} | "
                   f"mye_prob={mye_mean:.3f}±{mye_std:.3f}")
            if self.log_baseline:
                msg += (f" | baseline={self._baseline_mye:.3f} | "
                        f"lift={mye_mean - self._baseline_mye:+.3f}")
            msg += f" | {time.time() - t0:.1f}s"
            print(msg, file=self._stdout, flush=True)

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_eval:
            self._run_eval()
            self._next_eval += self.eval_freq
        return True

    def _on_training_end(self) -> None:
        # Final evaluation so the end-of-training point is always logged.
        self._run_eval()
