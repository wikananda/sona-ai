from dataclasses import dataclass
from typing import Optional
from copy import deepcopy

from sona_ai.core import resolve_device


TRANSCRIPTION_MODEL_PROFILES = {
    "parakeet": ("parakeet", "parakeet"),
    "faster-whisper-large-v3": ("faster_whisper", "faster-whisper-large-v3"),
    "faster-whisper-turbo": ("faster_whisper", "faster-whisper-turbo"),
}


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

    def to_metadata(self) -> dict:
        return {
            "transcription": {
                "engine": self.transcription_engine,
                "config": self.transcription_config,
            },
            "alignment": {
                "enabled": self.alignment_enabled,
                "engine": self.alignment_engine,
                "config": self.alignment_config,
            },
            "diarization": {
                "enabled": self.diarization_enabled,
                "engine": self.diarization_engine,
                "config": self.diarization_config,
            },
            "runtime": {
                "device": self.device,
                "requested_device": self.device,
                "resolved_device": resolve_device(self.device),
            },
        }

def resolve_pipeline_profile(
    speech_config: dict,
    model: Optional[str] = None,
    device: str= "auto",
    alignment_enabled: Optional[bool] = None,
    diarization_enabled: Optional[bool] = None,
) -> PipelineProfile:
    config = deepcopy(speech_config)

    transcription = config.get("transcription", {})
    alignment = config.get("alignment", {})
    diarization = config.get("diarization", {})

    requested_model = (model or transcription.get("engine", "parakeet")).lower()
    if requested_model in TRANSCRIPTION_MODEL_PROFILES:
        transcription_engine, transcription_config = TRANSCRIPTION_MODEL_PROFILES[requested_model]
    else:
        transcription_engine = requested_model
        transcription_config = transcription.get("config", transcription_engine)

    if alignment_enabled is None:
        resolved_alignment_enabled = bool(alignment.get("enabled", False))
    else:
        resolved_alignment_enabled = alignment_enabled
    alignment_engine = alignment.get("engine", "none").lower()
    alignment_config = alignment.get("config")

    if diarization_enabled is None:
        resolved_diarization_enabled = bool(diarization.get("enabled", True))
    else:
        resolved_diarization_enabled = diarization_enabled
    diarization_engine = diarization.get("engine", "community_external").lower()
    diarization_config = diarization.get("config")

    return PipelineProfile(
        transcription_engine=transcription_engine,
        transcription_config=transcription_config,
        alignment_enabled=resolved_alignment_enabled,
        alignment_engine=alignment_engine,
        alignment_config=alignment_config,
        diarization_enabled=resolved_diarization_enabled,
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
