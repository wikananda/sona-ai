import gc
from pathlib import Path
from typing import Optional

import torch
import whisperx

from sona_ai.core import PROJECT_ROOT, Timer, setup_logging
from sona_ai.transcription.schemas import TranscriptionResult


logger = setup_logging()


class WhisperXTranscriber:
    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.cache_dir = self._cache_dir()

    def load_models(self):
        logger.info("Loading WhisperX transcription model...")
        model_config = self.config["model"]
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.model = whisperx.load_model(
            model_config["whisper_model"],
            language=model_config["language"],
            device=model_config["device"],
            compute_type=model_config["compute_type"],
            download_root=str(self.cache_dir / "whisper"),
        )

    def _cache_dir(self) -> Path:
        cache_dir = self.config.get("cp_dir", {}).get("hf_cache", "cp/hf_cache")
        return PROJECT_ROOT / cache_dir

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        if self.model is None:
            raise ReferenceError("WhisperX transcription model is not initialized.")

        audio = whisperx.load_audio(audio_path)
        transcription = self._run_transcription(audio, language=language)
        return TranscriptionResult.from_whisperx_result(transcription)

    def _run_transcription(self, audio, language: Optional[str] = None) -> dict:
        logger.info("Running transcription...")
        with Timer("Transcription"):
            return self.model.transcribe(
                audio,
                batch_size=self.config["model"]["batch_size"],
                language=language or self.config["model"].get("language"),
            )

    def cleanup_models(self):
        if self.model is not None:
            del self.model

        self.model = None
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
