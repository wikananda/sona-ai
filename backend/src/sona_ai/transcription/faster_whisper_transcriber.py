from pathlib import Path
from typing import Optional

import torch

from sona_ai.core import (
    Timer,
    cleanup_model_attrs,
    model_cache_root,
    setup_logging,
    setup_model_cache_environment,
)
from sona_ai.transcription.schemas import TranscriptionResult


logger = setup_logging()


class FasterWhisperTranscriber:
    def __init__(self, config: dict):
        self.config = config
        model_config = self.config["model"]
        self.model = None
        self.model_name = model_config["model_name"]
        self.device = self._resolve_device(model_config.get("device", "auto"))
        self.compute_type = model_config.get("compute_type") or self._default_compute_type()
        self.cpu_threads = model_config.get("cpu_threads", 0)
        self.num_workers = model_config.get("num_workers", 1)
        self.language = model_config.get("language")
        self.task = model_config.get("task", "transcribe")
        self.beam_size = model_config.get("beam_size", 5)
        self.vad_filter = model_config.get("vad_filter", False)
        self.word_timestamps = model_config.get("word_timestamps", False)
        self.cache_dir = self._cache_dir()

    def load_models(self):
        logger.info(
            "Loading Faster-Whisper model: %s on %s (%s)",
            self.model_name,
            self.device,
            self.compute_type,
        )
        self._setup_cache_environment()

        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise ImportError(
                "Faster-Whisper transcription requires faster-whisper. "
                "Install it with `pip install -r backend/requirements.txt`."
            ) from exc

        self.model = WhisperModel(
            self.model_name,
            device=self.device,
            compute_type=self.compute_type,
            cpu_threads=self.cpu_threads,
            num_workers=self.num_workers,
            download_root=str(self.cache_dir / "faster-whisper"),
        )

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        if self.model is None:
            raise ReferenceError("Faster-Whisper transcription model is not initialized.")

        resolved_language = language or self.language
        logger.info("Running Faster-Whisper transcription...")
        with Timer("Faster-Whisper transcription"):
            segments, info = self.model.transcribe(
                audio_path,
                language=resolved_language,
                task=self.task,
                beam_size=self.beam_size,
                vad_filter=self.vad_filter,
                word_timestamps=self.word_timestamps,
            )
            segments = list(segments)

        return TranscriptionResult.from_faster_whisper_result(
            segments,
            info,
            language=resolved_language,
        )

    def _resolve_device(self, device: str) -> str:
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            return "cpu"
        if device == "mps":
            logger.warning("Faster-Whisper does not support MPS. Falling back to CPU.")
            return "cpu"
        return device

    def _default_compute_type(self) -> str:
        if self.device == "cuda":
            return "float16"
        return "int8"

    def _cache_dir(self) -> Path:
        return model_cache_root(self.config)

    def _setup_cache_environment(self):
        self.cache_dir = setup_model_cache_environment(self.config)

    def cleanup_models(self):
        cleanup_model_attrs(self, "model")
