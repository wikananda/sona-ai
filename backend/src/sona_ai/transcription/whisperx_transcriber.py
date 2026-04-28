import gc
from typing import Optional

import torch
import whisperx

from sona_ai.core import Timer, setup_logging
from sona_ai.transcription.schemas import TranscriptionResult


logger = setup_logging()


class WhisperXTranscriber:
    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.align_model = None
        self.align_metadata = None

    def load_models(self):
        logger.info("Loading WhisperX transcription models...")
        model_config = self.config["model"]

        self.model = whisperx.load_model(
            model_config["whisper_model"],
            language=model_config["language"],
            device=model_config["device"],
            compute_type=model_config["compute_type"],
        )
        self.align_model, self.align_metadata = whisperx.load_align_model(
            language_code=model_config["language"],
            device=model_config["device"],
            model_name=model_config["align_model"],
        )

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        if self.model is None or self.align_model is None:
            raise ReferenceError("WhisperX transcription models are not initialized.")

        audio = whisperx.load_audio(audio_path)
        transcription = self._run_transcription(audio, language=language)
        aligned = self._run_alignment(transcription, audio)
        return TranscriptionResult.from_whisperx_result(aligned)

    def _run_transcription(self, audio, language: Optional[str] = None) -> dict:
        logger.info("Running transcription...")
        with Timer("Transcription"):
            return self.model.transcribe(
                audio,
                batch_size=self.config["model"]["batch_size"],
                language=language or self.config["model"].get("language"),
            )

    def _run_alignment(self, result: dict, audio) -> dict:
        logger.info("Running alignment...")
        with Timer("Alignment"):
            return whisperx.align(
                result["segments"],
                self.align_model,
                self.align_metadata,
                audio,
                device=self.config["model"]["device"],
                return_char_alignments=False,
            )

    def cleanup_models(self):
        for model in [self.model, self.align_model]:
            if model is not None:
                del model

        self.model = None
        self.align_model = None
        self.align_metadata = None
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

