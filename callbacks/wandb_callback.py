"""
Custom callbacks for training.
"""
import time
import wandb
from stable_baselines3.common.callbacks import BaseCallback


class WandbCallback(BaseCallback):
    """
    Custom callback that logs PPO training metrics to Weights & Biases in real-time.
    """
    
    def __init__(self, verbose=0):
        """
        Initialize callback.
        
        Args:
            verbose (int): Verbosity level
        """
        super(WandbCallback, self).__init__(verbose)
        self.num_timesteps_last_logged = 0

    def _on_step(self) -> bool:
        """Called after every step in the environment."""
        
        # Only log if W&B is initialized
        if wandb.run is None:
            return True
        
        # Log to W&B every 100 timesteps
        if self.num_timesteps >= self.num_timesteps_last_logged + 100:
            log_dict = {}
            
            # Basic timestep info
            log_dict['timesteps'] = self.num_timesteps
            
            # Episode info
            if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
                latest_ep = self.model.ep_info_buffer[-1]
                log_dict['episode_reward'] = latest_ep.get('r', 0)
                log_dict['episode_length'] = latest_ep.get('l', 0)
                log_dict['episodes'] = len(self.model.ep_info_buffer)
            
            # Model logger metrics
            if hasattr(self.model, 'logger') and self.model.logger is not None:
                if hasattr(self.model.logger, 'name_to_value'):
                    mean_dict = self.model.logger.name_to_value
                    if mean_dict:
                        for key, value in mean_dict.items():
                            if isinstance(value, (int, float)):
                                clean_key = key.replace('/', '_')
                                log_dict[clean_key] = value
            
            # FPS calculation
            if hasattr(self.model, 'start_time') and self.model.start_time:
                elapsed = time.time() - self.model.start_time
                if elapsed > 0:
                    log_dict['fps'] = int(self.num_timesteps / elapsed)
            
            # Log if we have data
            if len(log_dict) > 1:
                wandb.log(log_dict)
                if self.verbose > 0:
                    print(f"[WandbCallback] Logged metrics at {self.num_timesteps} timesteps")
            
            self.num_timesteps_last_logged = self.num_timesteps
        
        return True

    def _on_training_end(self) -> None:
        """Called when training ends."""
        # Only log if W&B is initialized
        if wandb.run is None:
            return
        
        log_dict = {
            'training_complete': True,
            'final_timesteps': self.num_timesteps
        }
        wandb.log(log_dict, step=self.num_timesteps)
        if self.verbose > 0:
            print(f"[WandbCallback] Training completed! Final timesteps: {self.num_timesteps}")
