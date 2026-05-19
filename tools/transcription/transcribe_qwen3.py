from sympy.utilities.misc import strlines
import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from dotenv import load_dotenv
from qwen_asr import Qwen3ASRModel

DEFAULT_MODEL = "Qwen/Qwen3-ASR-0.6B"

LANGUAGE_ALIASES = {
    "id": "Indonesian",
    "indonesian": "Indonesian",
    "indonesia": "Indonesian",
    "en": "English",
    "english": "English",
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen3-ASR transcription.")
    parser.add_argument("audio_path", help="Path to input audio.")
    parser.add_argument("output_path", help="Path to write transcription JSON.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--language", default=None, help="Use id, en, Indonesian, English, or omit for auto.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    return parser.parse_args()

def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def resolve_dtype(dtype: str, device: str):
    if dtype == "float32":
        return torch.float32
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if device == "cuda":
        return torch.bfloat16
    return torch.float32

def resolve_device_map(device: str) -> str:
    if device == "cuda":
        return "cuda:0"
    return device

def resolve_language(language: str | None) -> str | None:
    if not language:
        return None
    normalized = language.strip().lower()
    return LANGUAGE_ALIASES.get(normalized, normalized)

def result_value(result: Any, key: str, default=None):
    if isinstance(result, dict):
        return result.get(key, default)
    return getattr(result, key, default)

def normalize_result(result: Any, requested_language: str | None) -> dict:
    text = str(result_value(result, "text", "") or "")
    language = result_value(result, "language", requested_language)
    raw = result if isinstance(result, dict) else {
        "text": text,
        "language": language,
        "time_stamps": result_value(result, "time_stamps", None)
    }

    return {
        "text": text,
        "language": language,
        "segments": [
            {
                "text": text,
                "start": 0.0,
                "end": 0.0,
                "words": [],
            }
        ],
        "raw": raw,
    }

def main() -> None:
    args = parse_args()
    repo_root = Path.cwd()
    load_dotenv(repo_root / ".env")

    audio_path = Path(args.audio_path)
    output_path = Path(args.output_path)

    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    language = resolve_language(args.language)

    model = Qwen3ASRModel.from_pretrained(
        args.model,
        dtype=dtype,
        device_map=resolve_device_map(device),
        max_new_tokens=args.max_new_tokens,
    )

    results = model.transcribe(
        audio=str(audio_path),
        language=language,
    )

    first_result = results[0] if isinstance(results, list) else results
    output = normalize_result(first_result, language)

    with output_path.open("w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Wrote transcription to {output_path}")
    print(f"Language: {output.get('language')}")
    print(output.get("text", "")[:500])

if __name__ == "__main__":
    main()