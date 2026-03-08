"""
Weights & Biases configuration.
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class WandbConfig:
    """Weights & Biases integration configuration."""
    
    # Project and run settings
    project: str = "enso-rl"
    entity: str = "ayms-university-of-maryland"  # Change to your wandb username
    name: str = "ppo-enso-training"
    notes: str = "Training ENSO control using PPO with continuous environment (no episodic resets)"
    
    # Run configuration
    tags: list = field(default_factory=lambda: ["enso", "reinforcement-learning", "ppo", "climate-control"])
    mode: str = "online"  # "online", "offline", or "disabled"
    api_key: Optional[str] = None  # Set from environment if None
    
    # Logging frequency
    log_interval: int = 100  # timesteps between W&B logs
    
    # Model artifact config
    save_model_artifact: bool = True
    artifact_type: str = "model"
    
    # Advanced features
    enable_media_logging: bool = True
    enable_custom_charts: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for W&B initialization."""
        return {
            "project": self.project,
            "entity": self.entity,
            "name": self.name,
            "notes": self.notes,
            "tags": self.tags,
            "mode": self.mode,
            "group": "training",
            "job_type": "training",
        }
