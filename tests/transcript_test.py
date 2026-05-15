import argparse
import os
from typing import Any, Optional

from sona_ai.core import load_config
from sona_ai.pipelines import build_speech_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the configured speech pipeline on a local audio file.",
    )
    parser.add_argument("--audio-path", default=None)
    parser.add_argument("--language", default=None)
    parser.add_argument("--min-speakers", type=int, default=None)
    parser.add_argument("--max-speakers", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def config_value(config: dict[str, Any], *keys: str) -> Optional[Any]:
    current: Any = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def env_int(name: str) -> Optional[int]:
    value = os.getenv(name)
    return int(value) if value else None


def resolve_engine_config(speech_config: dict[str, Any]) -> dict[str, Any]:
    transcription_config = speech_config.get("transcription", {})
    engine = os.getenv(
        "SONA_TRANSCRIPTION_ENGINE",
        transcription_config.get("engine", "whisperx"),
    )
    config_name = os.getenv(
        "SONA_TRANSCRIPTION_CONFIG",
        transcription_config.get("config", engine),
    )
    return load_config(config_name)


def main() -> None:
    args = parse_args()
    speech_config = load_config("speech")
    engine_config = resolve_engine_config(speech_config)

    audio_path = (
        args.audio_path
        or os.getenv("SONA_TEST_AUDIO")
        or config_value(speech_config, "input", "audio_file")
        or config_value(engine_config, "input", "audio_file")
        or "data/raw/audio/audio.mp3"
    )
    language = (
        args.language
        or os.getenv("SONA_TEST_LANGUAGE")
        or config_value(speech_config, "input", "language")
    )
    min_speakers = (
        args.min_speakers
        or env_int("SONA_TEST_MIN_SPEAKERS")
        or config_value(speech_config, "input", "min_speakers")
        or config_value(engine_config, "input", "min_speakers")
    )
    max_speakers = (
        args.max_speakers
        or env_int("SONA_TEST_MAX_SPEAKERS")
        or config_value(speech_config, "input", "max_speakers")
        or config_value(engine_config, "input", "max_speakers")
    )

    pipeline = build_speech_pipeline(
        speech_config,
        device=args.device or os.getenv("SONA_TEST_DEVICE"),
    )
    pipeline.load_models()

    try:
        result = pipeline.transcribe(
            str(audio_path),
            language=language,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )

        print("\nTranscript preview:\n")
        for segment in result["transcript"][:20]:
            print(
                f"[{segment['start']:.2f} - {segment['end']:.2f}] "
                f"{segment['speaker']}: {segment['text']}"
            )

        print("\nSaved outputs:")
        print("outputs/transcription/conversations.json")
        print("outputs/transcription/result_raw.json")

    finally:
        print("\nCleaning up...")
        pipeline.cleanup_models()


if __name__ == "__main__":
    main()
