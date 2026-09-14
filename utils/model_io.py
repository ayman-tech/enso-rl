"""Shared model-loading utility used by analysis scripts."""
import sys
import json
import warnings
from dataclasses import asdict, fields
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from stable_baselines3 import PPO
from config import EnvConfig
from utils.data_processing import load_observational_data, prepare_xro_parameters
from envs import XROMultiYearEnv
from XRO.core import XRO

# Written next to models/<name>.zip by scripts/train.py.
ENVCONFIG_SUFFIX = "_envconfig.json"


def _resolve_model_path(model_path: str) -> Path:
    """Normalise a model name/path to models/<name>.zip."""
    s = str(model_path)
    if not s.endswith('.zip'):
        s += '.zip'
    if not s.startswith('models'):
        s = f'models/{s}'
    return Path(s)


def envconfig_path(model_path: str) -> Path:
    """Sidecar config path for a model: models/<name>_envconfig.json."""
    p = _resolve_model_path(model_path)
    return p.with_name(p.stem + ENVCONFIG_SUFFIX)


def save_env_config(model_path: str, env_config: EnvConfig) -> Path:
    """Write the EnvConfig a model was trained under, beside its .zip.

    Without this, the only record of a run's regime / clip_mode / bounds_scale /
    action_scale is whatever EnvConfig happens to default to when someone later runs
    inference -- which is how model10 became unreproducible after the defaults moved.
    """
    out = envconfig_path(model_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(env_config), indent=2, sort_keys=True))
    return out


def resolve_env_config(model_path: str, base: EnvConfig = None) -> EnvConfig:
    """The EnvConfig a model was trained under: its sidecar if present, else `base`.

    Falls back with a warning rather than an error, so models predating the sidecar
    (model4..model12) still load -- but the warning is the cue that clip_mode /
    bounds_scale / regime must be supplied by hand for those.
    """
    path = envconfig_path(model_path)
    if not path.exists():
        warnings.warn(
            f"No {path.name} beside {Path(model_path).name} -- falling back to the "
            f"current EnvConfig defaults. Pass --clip-mode/--bounds-scale/--regime "
            f"explicitly if this model was trained under different settings.",
            stacklevel=2)
        return base if base is not None else EnvConfig()

    data = json.loads(path.read_text())
    known = {f.name for f in fields(EnvConfig)}
    unknown = set(data) - known
    if unknown:
        # A config written by a newer revision than this checkout. Dropping keys
        # silently would reintroduce exactly the bug this file exists to prevent.
        raise ValueError(
            f"{path} has fields unknown to this EnvConfig: {sorted(unknown)}. "
            f"The checkout is older than the model.")
    return EnvConfig(**data)


def load_environment(model_path: str, env_config: EnvConfig):
    """Load a trained PPO model and its paired XROMultiYearEnv.

    Args:
        model_path: Model name or path (with or without .zip / models/ prefix).
        env_config: Environment configuration.

    Returns:
        (model, env, var_names)
    """
    p = _resolve_model_path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"Model not found: {p}")

    obs_ds, train_ds, var_names, bounds = load_observational_data(
        env_config.data_config["data_path"],
        env_config.data_config["train_start"],
        env_config.data_config["train_end"],
    )

    model_xro = XRO()
    params = prepare_xro_parameters(model_xro, train_ds, var_names,
                                    config=env_config)
    params['threshold'] = env_config.threshold

    env = XROMultiYearEnv(
        params=params, train_ds=train_ds,
        var_names=var_names, max_steps=env_config.max_steps,
    )

    model = PPO.load(str(p), env=env)
    return model, env, var_names
