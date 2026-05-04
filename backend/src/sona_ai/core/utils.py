import json
import logging
import math
from pathlib import Path
from typing import Any, Union

import numpy as np
import yaml

from .paths import PROJECT_ROOT


def setup_logging():
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)
    return logging.getLogger(__name__)


def sanitize_for_json(obj):
    """
    Make JSON safe and ready output.
    Result from whisperx may contain numpy data or NaN type which is not JSON serializable.
    This function will convert numpy data to Python data and NaN to None.

    obj: object to make JSON safe (result['segments'])
    """

    # Doing recursion, because might encounter nested dict/list/numpy data inside the obj/dict/list
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, np.generic):
        value = obj.item()
        if isinstance(value, float) and math.isnan(value):
            return None
        return value
    elif _is_torch_tensor(obj):
        if obj.numel() == 1:
            return sanitize_for_json(obj.item())
        return sanitize_for_json(obj.detach().cpu().tolist())
    elif isinstance(obj, float):
        if math.isnan(obj):
            return None
        return obj
    else:
        return obj


def _is_torch_tensor(obj):
    try:
        import torch
    except ImportError:
        return False

    return isinstance(obj, torch.Tensor)


def write_json(path: Union[str, Path], data: Any):
    """
    Write JSON data to a file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(path: Union[str, Path]):
    """
    Load JSON data from a file
    """
    with open(path, 'r') as f:
        return json.load(f)


def load_config(config: Union[str, dict]):
    """
    Load config from a YAML file or return the dict if already loaded.
    config: str or dict -> name of the config, path to YAML, or the config dict itself.
    """
    if isinstance(config, dict):
        return config

    # Try to find the config file
    config_path = Path(config)
    
    # If not a direct path, look in the configs directory
    if not config_path.exists():
        potential_path = PROJECT_ROOT / 'configs' / f"{config}.yaml"
        if potential_path.exists():
            config_path = potential_path
        else:
            # Try without .yaml extension if it was already provided in config
            potential_path = PROJECT_ROOT / 'configs' / config
            if potential_path.exists():
                config_path = potential_path
            else:
                raise ValueError(f"Config not found: {config}. Checked path and configs/ directory.")
        
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
