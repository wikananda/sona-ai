import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download WhisperX Wav2Vec2 aligner models.")
    parser.add_argument("--model-name", action="append", required=True)
    parser.add_argument("--cache-dir", default=".models/wav2vec2-align")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    import whisperx

    for model_name in args.model_name:
        language = "id" if "indonesian" in model_name.lower() else "en"
        whisperx.load_align_model(
            language_code=language,
            device=args.device,
            model_name=model_name,
            model_dir=str(cache_dir),
        )
        print(f"Downloaded Wav2Vec2 aligner model: {model_name}")


if __name__ == "__main__":
    main()
