import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import torch
from dotenv import load_dotenv
from pyannote.audio import Pipeline

MODEL_NAME = "pyannote/speaker-diarization-community-1"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pyannote Community-1 diarization.")
    parser.add_argument("audio_path", help="Path to the input audio file.")
    parser.add_argument("output_path", help="Path to write diarization JSON.")
    parser.add_argument("--min-speakers", type=int, default=None)
    parser.add_argument("--max-speakers", type=int, default=None)
    parser.add_argument("--num-speakers", type=int, default=None)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--cache-dir", default=".models/pyannote-community")
    parser.add_argument("--regular", action="store_true", help="Use regular diarization instead of exclusive diarization.")
    return parser.parse_args()

def normalize_speaker(label: object) -> str:
    label_text = str(label)
    if label_text.startswith("SPEAKER_"):
        return label_text
    if label_text.isdigit():
        return f"SPEAKER_{int(label_text):02d}"
    return label_text

def annotation_to_turns(annotation) -> list[dict]:
    turns = []
    for segment, _, speaker in annotation.itertracks(yield_label=True):
        turns.append(
            {
                "start": float(segment.start),
                "end": float(segment.end),
                "speaker": normalize_speaker(speaker)
            }
        )
    return turns

def normalize_audio(input_path: Path) -> Path:
    output_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    output_path = Path(output_file.name)
    output_file.close()

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]

    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required to normalize audio before diarization.") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg failed to normalize audio: {input_path}") from exc

    return output_path

def main() -> None:
    args = parse_args()
    repo_root = Path.cwd()
    load_dotenv(repo_root / ".env")

    token = os.getenv("HF_TOKEN")
    if not token:
        raise EnvironmentError("HF_TOKEN not set in .env")
    
    audio_path = Path(args.audio_path)
    output_path = Path(args.output_path)
    cache_dir = Path(args.cache_dir)

    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pipeline = Pipeline.from_pretrained(
        MODEL_NAME,
        token=token,
        cache_dir=cache_dir,
    ).to(torch.device(args.device))

    diarization_kwargs = {}
    if args.num_speakers is not None:
        diarization_kwargs["num_speakers"] = args.num_speakers
    else:
        if args.min_speakers is not None:
            diarization_kwargs["min_speakers"] = args.min_speakers
        if args.max_speakers is not None:
            diarization_kwargs["max_speakers"] = args.max_speakers
    
    normalized_audio_path = normalize_audio(audio_path)
    try:
        output = pipeline(str(normalized_audio_path), **diarization_kwargs)
    finally:
        normalized_audio_path.unlink(missing_ok=True)

    if args.regular:
        annotation = getattr(output, "speaker_diarization", output)
    else:
        annotation = getattr(output, "exclusive_speaker_diarization", output)

    turns = annotation_to_turns(annotation)

    with output_path.open("w") as f:
        json.dump(turns, f, indent=2)

    speakers = sorted({turn["speaker"] for turn in turns})
    print(f"Wrote {len(turns)} turns with {len(speakers)} speakers: {speakers}")

if __name__ == "__main__":
    main()
