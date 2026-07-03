import gc

import torch


def release_torch_memory() -> None:
    """Run garbage collection and empty the active torch device cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def cleanup_model_attrs(obj, *attr_names: str, empty_torch_caches: bool = True) -> None:
    """Null out model attributes on `obj`, then collect garbage.

    When `empty_torch_caches` is True (the default) the CUDA/MPS cache is also
    emptied after collection.
    """
    for name in attr_names:
        if getattr(obj, name, None) is not None:
            setattr(obj, name, None)

    if empty_torch_caches:
        release_torch_memory()
    else:
        gc.collect()
