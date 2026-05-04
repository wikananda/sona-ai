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
        self.align_models = {}
        self.cache_dir = self._cache_dir()

    def load_models(self):
        logger.info("Wav2Vec2 alignment models will load per language on first use.")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def align(
        self,
        transcription: TranscriptionResult,
        audio_path: str,
    ) -> TranscriptionResult:
        language = self._resolve_language(transcription.language)
        align_model, align_metadata = self._get_align_model(language)

        audio = whisperx.load_audio(audio_path)

        logger.info("Running alignment with %s wav2vec2 model...", language)
        with Timer("Alignment"):
            aligned = whisperx.align(
                transcription.to_segment_dicts(),
                align_model,
                align_metadata,
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

    def _get_align_model(self, language: str):
        if language in self.align_models:
            return self.align_models[language]

        model_name = self._align_model_name(language)
        logger.info("Loading %s alignment model: %s", language, model_name)
        align_model, align_metadata = whisperx.load_align_model(
            language_code=language,
            device=self.config["model"]["device"],
            model_name=model_name,
            model_dir=str(self.cache_dir),
        )
        self.align_models[language] = (align_model, align_metadata)
        return self.align_models[language]

    def _align_model_name(self, language: str) -> str:
        model_config = self.config["model"]
        align_models = model_config.get("align_models", {})
        return align_models.get(language) or model_config["align_model"]

    def _resolve_language(self, language: str | None) -> str:
        resolved = (language or self.config["model"].get("language") or "en").lower()
        aliases = {
            "eng": "en",
            "english": "en",
            "indonesian": "id",
            "indonesia": "id",
            "ind": "id",
        }
        return aliases.get(resolved, resolved)

    def cleanup_models(self):
        for align_model, _ in self.align_models.values():
            del align_model

        self.align_models = {}
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
