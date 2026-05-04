import os
from copy import deepcopy
from typing import Optional

from sona_ai.core import load_config, resolve_device
from sona_ai.pipelines.speaker_assignment import WhisperXSpeakerAssigner
from sona_ai.pipelines.speech_pipeline import SpeechPipeline


def build_speech_pipeline(
    config: Optional[dict] = None,
    engine: Optional[str] = None,
    engine_config_name: Optional[str] = None,
    device: Optional[str] = None,
    write_outputs: bool = True,
) -> SpeechPipeline:
    speech_config = deepcopy(config or load_config("speech"))
    transcription_config = speech_config.get("transcription", {})
    engine = (engine or os.getenv(
        "SONA_TRANSCRIPTION_ENGINE",
        transcription_config.get("engine", "whisperx"),
    )).lower()
    engine_config_name = engine_config_name or os.getenv(
        "SONA_TRANSCRIPTION_CONFIG",
        transcription_config.get("config", engine),
    )

    engine_config = deepcopy(load_config(engine_config_name))
    if device is not None:
        resolved_device = resolve_device(device)
        engine_config.setdefault("model", {})["device"] = resolved_device
        _set_diarization_device(speech_config, resolved_device)
    SpeechPipeline.setup_environment(config=engine_config)

    if engine == "whisperx":
        from sona_ai.alignment import Wav2Vec2Aligner
        from sona_ai.transcription import WhisperXTranscriber

        transcriber = WhisperXTranscriber(engine_config)
        aligner = Wav2Vec2Aligner(engine_config)
    elif engine == "parakeet":
        from sona_ai.transcription import ParakeetTranscriber

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
        write_outputs=write_outputs,
    )


def _build_diarizer(speech_config: dict):
    diarization_config = speech_config.get("diarization", {})
    if not diarization_config.get("enabled", True):
        return None

    config_name = diarization_config.get("config", "whisperx")
    from sona_ai.diarization import PyannoteDiarizer

    diarization_engine_config = deepcopy(load_config(config_name))
    device = diarization_config.get("device")
    if device is not None:
        diarization_engine_config.setdefault("model", {})["device"] = device
    return PyannoteDiarizer(diarization_engine_config)


def _set_diarization_device(speech_config: dict, device: str) -> None:
    speech_config.setdefault("diarization", {})["device"] = device
