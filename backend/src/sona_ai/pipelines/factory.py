import os
from typing import Optional

from sona_ai.alignment import Wav2Vec2Aligner
from sona_ai.core import load_config
from sona_ai.diarization import PyannoteDiarizer
from sona_ai.pipelines.speaker_assignment import WhisperXSpeakerAssigner
from sona_ai.pipelines.speech_pipeline import SpeechPipeline
from sona_ai.transcription import ParakeetTranscriber, WhisperXTranscriber


def build_speech_pipeline(config: Optional[dict] = None) -> SpeechPipeline:
    speech_config = config or load_config("speech")
    transcription_config = speech_config.get("transcription", {})
    engine = os.getenv(
        "SONA_TRANSCRIPTION_ENGINE",
        transcription_config.get("engine", "whisperx"),
    ).lower()
    engine_config_name = os.getenv(
        "SONA_TRANSCRIPTION_CONFIG",
        transcription_config.get("config", engine),
    )

    engine_config = load_config(engine_config_name)
    SpeechPipeline.setup_environment(config=engine_config)

    if engine == "whisperx":
        transcriber = WhisperXTranscriber(engine_config)
        aligner = Wav2Vec2Aligner(engine_config)
    elif engine == "parakeet":
        transcriber = ParakeetTranscriber(engine_config)
        aligner = None
    else:
        raise ValueError(f"Unsupported transcription engine: {engine}")

    diarizer = _build_diarizer(speech_config)

    return SpeechPipeline(
        transcriber=transcriber,
        aligner=aligner,
        diarizer=diarizer,
        speaker_assigner=WhisperXSpeakerAssigner(),
    )


def _build_diarizer(speech_config: dict):
    diarization_config = speech_config.get("diarization", {})
    if not diarization_config.get("enabled", True):
        return None

    config_name = diarization_config.get("config", "whisperx")
    return PyannoteDiarizer(load_config(config_name))

