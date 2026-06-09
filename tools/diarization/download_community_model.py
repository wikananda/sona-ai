import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

MODEL_NAME = "pyannote/speaker-diarization-community-1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download pyannote Community-1 diarization model.")
    parser.add_argument("--cache-dir", default=".models/pyannote-community/pyannote-community")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd()
    load_dotenv(repo_root / ".env")

    token = os.getenv("HF_TOKEN")
    if not token:
        raise EnvironmentError("HF_TOKEN not set in .env")

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    from pyannote.audio import Pipeline

    Pipeline.from_pretrained(
        MODEL_NAME,
        token=token,
        cache_dir=cache_dir,
    )
    print(f"Downloaded pyannote diarization model: {MODEL_NAME}")


if __name__ == "__main__":
    main()
