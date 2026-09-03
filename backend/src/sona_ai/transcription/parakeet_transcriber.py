import gc
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from sona_ai.core import Timer, model_cache_root, setup_logging, setup_model_cache_environment
from sona_ai.transcription.schemas import TranscriptionResult


logger = setup_logging()


class ParakeetTranscriber:
    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.model_name = self.config["model"]["model_name"]
        self.device = self._resolve_device(self.config["model"].get("device", "auto"))
        self.language = self.config["model"].get("language")
        self.supported_languages = set(self.config["model"].get("supported_languages", ["en"]))
        self.batch_size = self.config["model"].get("batch_size")
        self.cache_dir = self._cache_dir()

    def load_models(self):
        logger.info("Loading Parakeet transcription model...")
        self._setup_cache_environment()
        self._patch_numpy_sctypes()

        try:
            import nemo.collections.asr as nemo_asr
        except ImportError as exc:
            raise ImportError(
                "Parakeet transcription requires NVIDIA NeMo. "
                "Install it with `pip install -r backend/requirements.txt`."
            ) from exc
        except RuntimeError as exc:
            if "torchvision::nms" in str(exc):
                raise RuntimeError(
                    "Parakeet could not import NeMo because torchvision is not "
                    "compatible with the installed torch. This project pins "
                    "torch==2.8.0, which needs torchvision==0.23.0. Run "
                    "`pip install --upgrade torchvision==0.23.0` or reinstall "
                    "`backend/requirements.txt` after pulling this change."
                ) from exc
            raise

        self.model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=self.model_name,
        )
        if hasattr(self.model, "to"):
            self.model = self.model.to(self.device)
        if hasattr(self.model, "eval"):
            self.model.eval()

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        if self.model is None:
            raise ReferenceError("Parakeet transcription model is not initialized.")

        self._patch_numpy_sctypes()
        resolved_language = language or self.language
        self._validate_language(resolved_language)

        logger.info("Running Parakeet transcription...")
        with Timer("Parakeet transcription"):
            hypotheses = self._transcribe_with_timestamps(audio_path)

        return TranscriptionResult.from_parakeet_hypothesis(
            self._first_hypothesis(hypotheses),
            language=resolved_language,
        )

    def transcribe_samples(
        self,
        samples: np.ndarray,
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """Transcribe mono 16 kHz float32 samples without container conversion."""
        if self.model is None:
            raise ReferenceError("Parakeet transcription model is not initialized.")

        audio = np.asarray(samples, dtype=np.float32)
        if audio.ndim != 1:
            raise ValueError("Parakeet live audio must be a one-dimensional mono signal.")
        if not np.isfinite(audio).all():
            raise ValueError("Parakeet live audio contains non-finite samples.")
        if audio.size == 0:
            return TranscriptionResult(segments=[], language=language or self.language)

        self._patch_numpy_sctypes()
        resolved_language = language or self.language
        self._validate_language(resolved_language)
        hypotheses = self._transcribe_with_timestamps(audio, is_sample_array=True)
        return TranscriptionResult.from_parakeet_hypothesis(
            self._first_hypothesis(hypotheses),
            language=resolved_language,
        )

    def _transcribe_with_timestamps(
        self,
        audio,
        *,
        is_sample_array: bool = False,
    ):
        kwargs = {"timestamps": True, "verbose": False}
        if self.batch_size is not None:
            kwargs["batch_size"] = self.batch_size

        transcription_input = audio if is_sample_array else [audio]

        try:
            return self.model.transcribe(transcription_input, **kwargs)
        except TypeError:
            # Older NeMo releases don't accept every convenience keyword.
            kwargs.pop("verbose", None)
            try:
                return self.model.transcribe(transcription_input, **kwargs)
            except TypeError:
                kwargs.pop("batch_size", None)
                return self.model.transcribe(transcription_input, **kwargs)

    def _first_hypothesis(self, hypotheses):
        if hasattr(hypotheses, "text") or isinstance(hypotheses, (str, dict)):
            return hypotheses

        if isinstance(hypotheses, (list, tuple)):
            for item in hypotheses:
                try:
                    return self._first_hypothesis(item)
                except ValueError:
                    continue

        raise ValueError("Parakeet did not return a transcription hypothesis.")

    def _validate_language(self, language: Optional[str]):
        if not language or not self.supported_languages:
            return

        if language not in self.supported_languages:
            supported = ", ".join(sorted(self.supported_languages))
            raise ValueError(
                f"{self.model_name} does not support language={language!r}. "
                f"Supported languages: {supported}."
            )

    def _resolve_device(self, device: str) -> str:
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            return "cpu"
        return device

    def _cache_dir(self) -> Path:
        return model_cache_root(self.config)

    def _setup_cache_environment(self):
        self.cache_dir = setup_model_cache_environment(self.config)

    def _patch_numpy_sctypes(self):
        import numpy as np

        if hasattr(np, "sctypes"):
            return

        np.sctypes = {
            "int": [np.int8, np.int16, np.int32, np.int64],
            "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
            "float": [np.float16, np.float32, np.float64],
            "complex": [np.complex64, np.complex128],
            "others": [np.bool_, np.object_, np.bytes_, np.str_, np.void],
        }

    def cleanup_models(self):
        if self.model is not None:
            del self.model
            self.model = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
