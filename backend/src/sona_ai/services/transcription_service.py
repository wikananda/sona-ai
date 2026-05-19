import threading
from typing import Optional

from sona_ai.core import load_config, validate_device_available, setup_logging
from sona_ai.pipelines import SpeechPipeline, build_speech_pipeline
from sona_ai.services.pipeline_profile import (
    PipelineProfile,
    resolve_pipeline_profile,
    speech_config_for_profile,
)


SUPPORTED_TRANSCRIPTION_MODELS = {"parakeet"}

logger = setup_logging()


class TranscriptionService:
    def __init__(
        self,
        pipeline: SpeechPipeline,
        speech_config: Optional[dict] = None,
        default_model: str = "parakeet",
        default_device: str = "cpu",
    ):
        self.default_model = self._normalize_model(default_model)
        self.default_device = validate_device_available(default_device)
        self.speech_config = speech_config or load_config("speech")

        default_profile = resolve_pipeline_profile(
            self.speech_config,
            model=self.default_model,
            device=self.default_device,
        )
        self._pipelines = {
            default_profile.cache_key(): pipeline
        }

        self._pipeline_lock = threading.Lock()
        self._transcription_lock = threading.Lock()

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        model: Optional[str] = None,
        device: Optional[str] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ):
        pipeline = self._get_pipeline(model, device)
        with self._transcription_lock:
            return pipeline.transcribe(
                audio_path,
                language=language,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )

    def close(self):
        for pipeline in self._pipelines.values():
            pipeline.cleanup_models()

    def _get_pipeline(
        self,
        model: Optional[str],
        device: Optional[str],
    ) -> SpeechPipeline:
        profile = self.resolve_profile(model, device)
        key = profile.cache_key()
        logger.info(
            "Using speech pipeline: transcription=%s/%s alignment=%s/%s/%s "
            "diarization=%s/%s/%s device=%s",
            profile.transcription_engine,
            profile.transcription_config,
            profile.alignment_enabled,
            profile.alignment_engine,
            profile.alignment_config,
            profile.diarization_enabled,
            profile.diarization_engine,
            profile.diarization_config,
            profile.device,
        )
        if key in self._pipelines:
            return self._pipelines[key]

        with self._pipeline_lock:
            if key in self._pipelines:
                return self._pipelines[key]

            profile_config = speech_config_for_profile(self.speech_config, profile)
            pipeline = build_speech_pipeline(
                profile_config,
                device=profile.device,
            )
            pipeline.load_models()
            self._pipelines[key] = pipeline
            return pipeline

    def _normalize_model(self, model: str) -> str:
        model_name = model.lower().strip()
        if model_name not in SUPPORTED_TRANSCRIPTION_MODELS:
            allowed = ", ".join(sorted(SUPPORTED_TRANSCRIPTION_MODELS))
            raise ValueError(f"Unsupported transcription model: {model}. Use one of: {allowed}")
        return model_name

    def resolve_profile(
        self,
        model: Optional[str] = None,
        device: Optional[str] = None,
    ) -> PipelineProfile:
        model_name = self._normalize_model(model or self.default_model)
        device_name = validate_device_available(device or self.default_device)
        return resolve_pipeline_profile(
            self.speech_config,
            model=model_name,
            device=device_name,
        )
