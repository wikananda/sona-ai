from typing import Optional

import torch


SUPPORTED_DEVICES = {"auto", "cpu", "mps", "cuda"}


def runtime_devices() -> dict:
    available = ["auto", "cpu"]
    cuda_available = torch.cuda.is_available()
    mps_available = torch.backends.mps.is_available()

    if mps_available:
        available.append("mps")
    if cuda_available:
        available.append("cuda")

    return {
        "default": "auto",
        "available": available,
        "torch": {
            "cuda": cuda_available,
            "mps": mps_available,
        },
    }


def normalize_device(device: Optional[str]) -> str:
    device_name = (device or "auto").lower().strip()
    if device_name not in SUPPORTED_DEVICES:
        allowed = ", ".join(sorted(SUPPORTED_DEVICES))
        raise ValueError(f"Unsupported device: {device}. Use one of: {allowed}")
    return device_name


def resolve_device(device: Optional[str]) -> str:
    device_name = normalize_device(device)
    if device_name == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device_name


def validate_device_available(device: Optional[str]) -> str:
    device_name = normalize_device(device)
    if device_name == "auto":
        return device_name

    if device_name not in runtime_devices()["available"]:
        raise ValueError(f"Device is not available on this backend: {device_name}")
    return device_name
