"""Shared model-loading utility used by analysis scripts."""
import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from stable_baselines3 import PPO
from config import EnvConfig
from utils.data_processing import load_observational_data, prepare_xro_parameters
from envs import XROMultiYearEnv
from XRO.core import XRO


def load_environment(model_path: str, env_config: EnvConfig):
    """Load a trained PPO model and its paired XROMultiYearEnv.

    Args:
        model_path: Model name or path (with or without .zip / models/ prefix).
        env_config: Environment configuration.

    Returns:
        (model, env, var_names)
    """
    model_path_str = model_path
    if not model_path_str.endswith('.zip'):
        model_path_str += '.zip'
    if not model_path_str.startswith('models'):
        model_path_str = f'models/{model_path_str}'

    p = Path(model_path_str)
    if not p.exists():
        raise FileNotFoundError(f"Model not found: {p}")

    obs_ds, train_ds, var_names, bounds = load_observational_data(
        env_config.data_config["data_path"],
        env_config.data_config["train_start"],
        env_config.data_config["train_end"],
    )

    model_xro = XRO()
    params = prepare_xro_parameters(model_xro, train_ds, var_names, bounds)
    params['threshold'] = env_config.threshold

    env = XROMultiYearEnv(
        params=params, train_ds=train_ds,
        var_names=var_names, max_steps=env_config.max_steps,
    )

    model = PPO.load(str(p), env=env)
    return model, env, var_names
