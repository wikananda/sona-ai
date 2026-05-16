from dataclasses import dataclass
from typing import Optional
from copy import deepcopy

from sona_ai.core import resolve_device

@dataclass(frozen=True)
class PipelineProfile:
    transcription_engine: str
    transcription_config: str

    alignment_enabled: bool
    alignment_engine: str
    alignment_config: Optional[str]

    diarization_enabled: bool
    diarization_engine: str
    diarization_config: Optional[str]

    device: str

    def cache_key(self) -> tuple:
        return (
            self.transcription_engine,
            self.transcription_config,
            self.alignment_enabled,
            self.alignment_engine,
            self.alignment_config,
            self.diarization_enabled,
            self.diarization_engine,
            self.diarization_config,
            resolve_device(self.device),
        )

def resolve_pipeline_profile(
    speech_config: dict,
    model: Optional[str] = None,
    device: str= "auto",
) -> PipelineProfile:
    config = deepcopy(speech_config)

    transcription = config.get("transcription", {})
    alignment = config.get("alignment", {})
    diarization = config.get("diarization", {})

    transcription_engine = (model or transcription.get("engine", "parakeet")).lower()
    transcription_config = transcription.get("config", transcription_engine)

    if model is not None:
        transcription_config = transcription_engine

    alignment_enabled = bool(alignment.get("enabled", False))
    alignment_engine = alignment.get("engine", "none").lower()
    alignment_config = alignment.get("config")

    diarization_enabled = bool(diarization.get("enabled", True))
    diarization_engine = diarization.get("engine", "community_external").lower()
    diarization_config = diarization.get("config")

    return PipelineProfile(
        transcription_engine=transcription_engine,
        transcription_config=transcription_config,
        alignment_enabled=alignment_enabled,
        alignment_engine=alignment_engine,
        alignment_config=alignment_config,
        diarization_enabled=diarization_enabled,
        diarization_engine=diarization_engine,
        diarization_config=diarization_config,
        device=device,
    )

def speech_config_for_profile(
    speech_config: dict,
    profile: PipelineProfile,
) -> dict:
    config = deepcopy(speech_config)

    config.setdefault("transcription", {})
    config["transcription"]["engine"] = profile.transcription_engine
    config["transcription"]["config"] = profile.transcription_config

    config.setdefault("alignment", {})
    config["alignment"]["enabled"] = profile.alignment_enabled
    config["alignment"]["engine"] = profile.alignment_engine
    config["alignment"]["config"] = profile.alignment_config

    config.setdefault("diarization", {})
    config["diarization"]["enabled"] = profile.diarization_enabled
    config["diarization"]["engine"] = profile.diarization_engine
    config["diarization"]["config"] = profile.diarization_config

    return config