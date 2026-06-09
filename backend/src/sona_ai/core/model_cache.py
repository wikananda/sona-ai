import os
from pathlib import Path
from typing import Mapping, Optional

from sona_ai.core.paths import PROJECT_ROOT


DEFAULT_MODEL_CACHE = ".models"


def model_cache_root(config: Optional[Mapping] = None) -> Path:
    cache_dir = os.getenv("SONA_HF_CACHE") or DEFAULT_MODEL_CACHE
    return (PROJECT_ROOT / cache_dir).resolve()


def setup_model_cache_environment(config: Optional[Mapping] = None) -> Path:
    cache_dir = model_cache_root(config)
    cache_dir.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HF_HUB_CACHE"] = str(cache_dir / "hub")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_dir / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir / "transformers")
    os.environ["TORCH_HOME"] = str(cache_dir / "torch")
    os.environ["PYANNOTE_CACHE"] = str(cache_dir / "pyannote")
    os.environ["NEMO_HOME"] = str(cache_dir / "nemo")
    os.environ["NEMO_CACHE_DIR"] = str(cache_dir / "nemo")
    os.environ["XDG_CACHE_HOME"] = str(cache_dir / "xdg")
    os.environ["MPLCONFIGDIR"] = str(cache_dir / "matplotlib")
    return cache_dir


def model_manifest_dir(config: Optional[Mapping] = None) -> Path:
    return model_cache_root(config) / ".sona_models"
