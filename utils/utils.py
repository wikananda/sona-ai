import json
import logging
from pathlib import Path
import yaml
from inspect import signature
from transformers import Seq2SeqTrainingArguments
import evaluate
import numpy as np
import math

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
    elif isinstance(obj, float):
        if math.isnan(obj):
            return None
        return obj
    else:
        return obj
def preview_dataset(dataset):
    """
    Preview the summarization dataset
    """
    for i in range(5):
        print(dataset[i])

def write_json(path, data):
    """
    Write JSON data to a file
    """
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def load_json(path):
    """
    Load JSON data from a file
    """
    with open(path, 'r') as f:
        return json.load(f)

def load_config(config_name: str):
    """
    Load config from a YAML file
    config_name: str -> name of the config to be loaded (not path)
    """
    if config_name == 'flan-t5':
        config_path = Path(__file__).parent.parent / 'configs' / 'flan-t5.yaml'
    elif config_name == 'whisperx':
        config_path = Path(__file__).parent.parent / 'configs' / 'whisperx.yaml'
    else:
        raise ValueError(f"Unknown config name: {config_name}")
        
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def filter_training_args(arg_dict):
    """
    Gather all arguments inside the .yaml file automatically
    """
    # Get the parameters
    sig = signature(Seq2SeqTrainingArguments).parameters
    valid_keys = set(sig.keys())

    filtered_args = {}
    unused = []

    for k, v in arg_dict.items():
        if k in valid_keys:
            filtered_args[k] = v
        else:
            unused.append(k)

    if unused:
        print(f"Warning: Unused arguments: {unused}")
    return filtered_args