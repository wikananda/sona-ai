import gc
from pathlib import Path

import torch
import whisperx

from sona_ai.core import PROJECT_ROOT, Timer, setup_logging
from sona_ai.transcription.schemas import TranscriptionResult


logger = setup_logging()


class Wav2Vec2Aligner:
    def __init__(self, config: dict):
        self.config = config
        self.align_model = None
        self.align_metadata = None
        self.cache_dir = self._cache_dir()

    def load_models(self):
        logger.info("Loading wav2vec2 alignment model...")
        model_config = self.config["model"]
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.align_model, self.align_metadata = whisperx.load_align_model(
            language_code=model_config["language"],
            device=model_config["device"],
            model_name=model_config["align_model"],
            model_dir=str(self.cache_dir),
        )

    def align(
        self,
        transcription: TranscriptionResult,
        audio_path: str,
    ) -> TranscriptionResult:
        if self.align_model is None:
            raise ReferenceError("Alignment model is not initialized.")

        audio = whisperx.load_audio(audio_path)

        logger.info("Running alignment...")
        with Timer("Alignment"):
            aligned = whisperx.align(
                transcription.to_segment_dicts(),
                self.align_model,
                self.align_metadata,
                audio,
                device=self.config["model"]["device"],
                return_char_alignments=False,
            )

        if transcription.language and "language" not in aligned:
            aligned["language"] = transcription.language

        return TranscriptionResult.from_whisperx_result(aligned)

    def _cache_dir(self) -> Path:
        cache_dir = self.config.get("cp_dir", {}).get("hf_cache", "cp/hf_cache")
        return PROJECT_ROOT / cache_dir

    def cleanup_models(self):
        if self.align_model is not None:
            del self.align_model

        self.align_model = None
        self.align_metadata = None
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

