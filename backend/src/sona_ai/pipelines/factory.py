import os
from copy import deepcopy
from typing import Optional

from sona_ai.core import load_config, resolve_device, setup_logging
from sona_ai.pipelines.speaker_assignment import SpeakerAssigner
from sona_ai.pipelines.speech_pipeline import SpeechPipeline

logger = setup_logging()

def build_speech_pipeline(
    config: Optional[dict] = None,
    engine: Optional[str] = None,
    engine_config_name: Optional[str] = None,
    device: Optional[str] = None,
    write_outputs: bool = True,
) -> SpeechPipeline:
    speech_config = deepcopy(config or load_config("speech"))
    transcriber, transcriber_config = _build_transcriber(
        speech_config=speech_config,
        engine=engine,
        engine_config_name=engine_config_name,
        device=device,
    )
    SpeechPipeline.setup_environment(config=transcriber_config)

    aligner = _build_aligner(speech_config, device=device)
    diarizer = _build_diarizer(speech_config)

    return SpeechPipeline(
        transcriber=transcriber,
        aligner=aligner,
        diarizer=diarizer,
        speaker_assigner=SpeakerAssigner(),
        write_outputs=write_outputs,
    )

def _build_transcriber(
    speech_config: dict,
    engine: Optional[str] = None,
    engine_config_name: Optional[str] = None,
    device: Optional[str] = None,
):
    transcription_config = speech_config.get("transcription", {})
    resolved_engine = (engine or os.getenv(
        "SONA_TRANSCRIPTION_ENGINE",
        transcription_config.get("engine", "parakeet"),
    )).lower()
    resolved_config_name = engine_config_name or os.getenv(
        "SONA_TRANSCRIPTION_CONFIG",
        transcription_config.get("config", resolved_engine),
    )

    if resolved_engine != "parakeet":
        raise ValueError(
            f"Unsupported transcription engine: {resolved_engine}. "
            "Use parakeet for transcription and the external alignment tool for Wav2Vec2 alignment."
        )

    engine_config = deepcopy(load_config(resolved_config_name))
    if device is not None:
        engine_config.setdefault("model", {})["device"] = resolve_device(device)
    logger.info(f"Transcription engine: {resolved_engine}")
    from sona_ai.transcription import ParakeetTranscriber
    return ParakeetTranscriber(engine_config), engine_config


def _build_diarizer(speech_config: dict):
    diarization_config = speech_config.get("diarization", {})
    if not diarization_config.get("enabled", True):
        return None

    config_name = diarization_config.get("config", "diarization-community")

    diarization_engine_config = deepcopy(load_config(config_name))
    device = diarization_config.get("device")
    if device is not None:
        diarization_engine_config.setdefault("model", {})["device"] = device

    engine = (
        diarization_config.get("engine")
        or diarization_engine_config.get("diarization", {}).get("engine")
        or "community_external"
    )
    logger.info(f"Diarization engine: {engine}")
    if engine == "community_external":
        from sona_ai.diarization import ExternalCommunityDiarizer
        return ExternalCommunityDiarizer(diarization_engine_config)

    if engine == "pyannote":
        from sona_ai.diarization import PyannoteDiarizer
        return PyannoteDiarizer(diarization_engine_config)

    raise ValueError(f"Unsupported diarization engine: {engine}")

def _build_aligner(speech_config: dict, device: Optional[str]):
    alignment_config = speech_config.get("alignment", {})
    if not alignment_config.get("enabled", False):
        return None

    engine = alignment_config.get("engine", "none").lower()
    logger.info(f"Alignment engine: {engine}")
    if engine == "none":
        from sona_ai.alignment import NoOpAligner

        return NoOpAligner(alignment_config)
    
    config_name = alignment_config.get("config")
    aligner_config = deepcopy(load_config(config_name)) if config_name else {}
    if device is not None:
        aligner_config.setdefault("model", {})["device"] = resolve_device(device)
    
    if engine in {"wav2vec2", "wav2vec2_external"}:
        from sona_ai.alignment import ExternalWav2Vec2Aligner
        return ExternalWav2Vec2Aligner(aligner_config)

    raise ValueError(f"Unsupported alignment engine: {engine}")

def _set_diarization_device(speech_config: dict, device: str) -> None:
    speech_config.setdefault("diarization", {})["device"] = device
