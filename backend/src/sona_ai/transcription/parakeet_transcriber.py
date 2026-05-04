import gc
import os
from pathlib import Path
from typing import Optional

import torch

from sona_ai.core import PROJECT_ROOT, Timer, setup_logging
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

    def _transcribe_with_timestamps(self, audio_path: str):
        kwargs = {"timestamps": True}
        if self.batch_size is not None:
            kwargs["batch_size"] = self.batch_size

        try:
            return self.model.transcribe([audio_path], **kwargs)
        except TypeError:
            kwargs.pop("batch_size", None)
            return self.model.transcribe([audio_path], **kwargs)

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
        cache_dir = self.config.get("cp_dir", {}).get("hf_cache", "cp/hf_cache")
        return PROJECT_ROOT / cache_dir

    def _setup_cache_environment(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(self.cache_dir)
        os.environ["HF_HUB_CACHE"] = str(self.cache_dir / "hub")
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(self.cache_dir / "hub")
        os.environ["TRANSFORMERS_CACHE"] = str(self.cache_dir / "transformers")
        os.environ["TORCH_HOME"] = str(self.cache_dir / "torch")
        os.environ["NEMO_HOME"] = str(self.cache_dir / "nemo")
        os.environ["NEMO_CACHE_DIR"] = str(self.cache_dir / "nemo")
        os.environ["XDG_CACHE_HOME"] = str(self.cache_dir / "xdg")

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
