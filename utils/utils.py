import json
import logging

def setup_logging():
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)
    return logging.getLogger(__name__)

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