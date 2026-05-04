import threading
from typing import Optional

from sona_ai.core import load_config
from sona_ai.pipelines import SpeechPipeline, build_speech_pipeline


SUPPORTED_TRANSCRIPTION_MODELS = {"whisperx", "parakeet"}


class TranscriptionService:
    def __init__(
        self,
        pipeline: SpeechPipeline,
        speech_config: Optional[dict] = None,
        default_model: str = "parakeet",
    ):
        self.default_model = self._normalize_model(default_model)
        self.speech_config = speech_config or load_config("speech")
        self._pipelines = {self.default_model: pipeline}
        self._pipeline_lock = threading.Lock()
        self._transcription_lock = threading.Lock()

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        model: Optional[str] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ):
        pipeline = self._get_pipeline(model)
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

    def _get_pipeline(self, model: Optional[str]) -> SpeechPipeline:
        model_name = self._normalize_model(model or self.default_model)
        if model_name in self._pipelines:
            return self._pipelines[model_name]

        with self._pipeline_lock:
            if model_name in self._pipelines:
                return self._pipelines[model_name]

            pipeline = build_speech_pipeline(
                self.speech_config,
                engine=model_name,
                engine_config_name=model_name,
            )
            pipeline.load_models()
            self._pipelines[model_name] = pipeline
            return pipeline

    def _normalize_model(self, model: str) -> str:
        model_name = model.lower().strip()
        if model_name not in SUPPORTED_TRANSCRIPTION_MODELS:
            allowed = ", ".join(sorted(SUPPORTED_TRANSCRIPTION_MODELS))
            raise ValueError(f"Unsupported transcription model: {model}. Use one of: {allowed}")
        return model_name
