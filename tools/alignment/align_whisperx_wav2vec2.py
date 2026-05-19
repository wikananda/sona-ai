import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run WhisperX-backed Wav2Vec2 alignment.")
    parser.add_argument("audio_path", help="Path to the input audio file.")
    parser.add_argument("transcription_path", help="Path to transcription JSON.")
    parser.add_argument("output_path", help="Path to write aligned transcription JSON.")
    parser.add_argument("--language", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--cache-dir", default="cp/hf_cache/wav2vec2-align")
    return parser.parse_args()


def to_jsonable(value):
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def main() -> None:
    args = parse_args()
    audio_path = Path(args.audio_path)
    transcription_path = Path(args.transcription_path)
    output_path = Path(args.output_path)
    cache_dir = Path(args.cache_dir)

    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not transcription_path.is_file():
        raise FileNotFoundError(f"Transcription file not found: {transcription_path}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with transcription_path.open("r") as f:
        transcription = json.load(f)

    import whisperx

    segments = transcription.get("segments", [])
    language = args.language.lower()
    align_model, align_metadata = whisperx.load_align_model(
        language_code=language,
        device=args.device,
        model_name=args.model_name,
        model_dir=str(cache_dir),
    )
    audio = whisperx.load_audio(str(audio_path))
    aligned = whisperx.align(
        segments,
        align_model,
        align_metadata,
        audio,
        device=args.device,
        return_char_alignments=False,
    )
    aligned.setdefault("language", transcription.get("language") or language)

    with output_path.open("w") as f:
        json.dump(to_jsonable(aligned), f, indent=2)

    print(f"Wrote aligned transcription to {output_path}")


if __name__ == "__main__":
    main()
