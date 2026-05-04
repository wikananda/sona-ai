import threading
from typing import Optional

from sona_ai.core import load_config, resolve_device, validate_device_available
from sona_ai.pipelines import SpeechPipeline, build_speech_pipeline


SUPPORTED_TRANSCRIPTION_MODELS = {"whisperx", "parakeet"}


class TranscriptionService:
    def __init__(
        self,
        pipeline: SpeechPipeline,
        speech_config: Optional[dict] = None,
        default_model: str = "parakeet",
        default_device: str = "auto",
    ):
        self.default_model = self._normalize_model(default_model)
        self.default_device = validate_device_available(default_device)
        self.speech_config = speech_config or load_config("speech")
        self._pipelines = {
            self._cache_key(self.default_model, self.default_device): pipeline
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
        model_name = self._normalize_model(model or self.default_model)
        device_name = validate_device_available(device or self.default_device)
        key = self._cache_key(model_name, device_name)
        if key in self._pipelines:
            return self._pipelines[key]

        with self._pipeline_lock:
            if key in self._pipelines:
                return self._pipelines[key]

            pipeline = build_speech_pipeline(
                self.speech_config,
                engine=model_name,
                engine_config_name=model_name,
                device=device_name,
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

    def _cache_key(self, model: str, device: str) -> tuple[str, str]:
        return (model, resolve_device(device))
