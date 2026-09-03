import gc
import math
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from sona_ai.core import Timer, model_cache_root, setup_logging, setup_model_cache_environment
from sona_ai.transcription.schemas import TranscriptSegment, TranscriptionResult, WordSegment


logger = setup_logging()
SAMPLE_RATE = 16000
_SENTENCE_END = re.compile(r"[.!?\u3002\uff01\uff1f][\"'\u201d\u2019)]*$")


class WhisperMpsTranscriber:
    """Run Hugging Face Whisper on Apple Silicon through PyTorch MPS."""

    def __init__(self, config: dict):
        self.config = config
        model_config = config["model"]
        self.pipeline = None
        self.model_name = model_config["model_name"]
        self.revision = model_config.get("revision")
        self.device = model_config.get("device", "mps").lower()
        self.dtype_name = model_config.get("dtype", "float16").lower()
        self.language = model_config.get("language")
        self.task = model_config.get("task", "transcribe")
        self.batch_size = max(1, int(model_config.get("batch_size", 4)))
        self.live_batch_size = max(1, int(model_config.get("live_batch_size", 1)))
        self.chunk_length_s = max(1, int(model_config.get("chunk_length_s", 30)))
        self.word_timestamps = bool(model_config.get("word_timestamps", True))
        self.attn_implementation = model_config.get("attn_implementation", "sdpa")
        self.warmup_seconds = max(0.0, float(model_config.get("warmup_seconds", 1.0)))
        self.live_silence_rms_threshold = max(
            0.0,
            float(model_config.get("live_silence_rms_threshold", 0.0005)),
        )
        self.cache_dir = self._cache_dir()

    def load_models(self) -> None:
        if self.pipeline is not None:
            return
        self._require_mps()
        self._setup_cache_environment()
        logger.info(
            "Loading Whisper model %s on Apple MPS (%s)",
            self.model_name,
            self.dtype_name,
        )

        try:
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
        except ImportError as exc:
            raise ImportError(
                "Whisper MPS transcription requires transformers and accelerate. "
                "Install them with `pip install -r backend/requirements.txt`."
            ) from exc

        dtype = self._torch_dtype()
        cache_dir = str(self.cache_dir / "transformers")
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_name,
            cache_dir=cache_dir,
            revision=self.revision,
            dtype=dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            attn_implementation=self.attn_implementation,
        )
        processor = AutoProcessor.from_pretrained(
            self.model_name,
            cache_dir=cache_dir,
            revision=self.revision,
        )
        try:
            model.to(torch.device("mps"))
            inference_pipeline = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                device=torch.device("mps"),
                dtype=dtype,
            )
            pipeline_device = getattr(inference_pipeline, "device", None)
            if _device_type(pipeline_device) != "mps":
                raise RuntimeError("Whisper pipeline did not initialize on the MPS device.")
            self.pipeline = inference_pipeline
            self._warm_up()
        except Exception:
            del model
            gc.collect()
            torch.mps.empty_cache()
            raise

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        return self._transcribe_input(
            audio_path,
            language=language,
            batch_size=self.batch_size,
        )

    def transcribe_samples(
        self,
        samples: np.ndarray,
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        normalized_samples = np.asarray(samples, dtype=np.float32)
        if normalized_samples.ndim != 1:
            raise ValueError("Live Whisper audio must be a one-dimensional sample array.")
        if not np.isfinite(normalized_samples).all():
            raise ValueError("Live Whisper audio contains non-finite samples.")
        if normalized_samples.size == 0 or _rms(normalized_samples) <= self.live_silence_rms_threshold:
            return TranscriptionResult(segments=[], language=language or self.language)
        audio = {
            "array": np.ascontiguousarray(normalized_samples),
            "sampling_rate": SAMPLE_RATE,
        }
        return self._transcribe_input(
            audio,
            language=language,
            batch_size=self.live_batch_size,
        )

    def _transcribe_input(
        self,
        audio: Any,
        *,
        language: Optional[str],
        batch_size: int,
    ) -> TranscriptionResult:
        if self.pipeline is None:
            raise ReferenceError("Whisper MPS transcription model is not initialized.")

        resolved_language = language or self.language
        generate_kwargs = {"task": self.task}
        if resolved_language:
            generate_kwargs["language"] = resolved_language

        logger.info("Running Whisper transcription on Apple MPS...")
        inference_kwargs = {
            "chunk_length_s": self.chunk_length_s,
            "batch_size": batch_size,
            "return_timestamps": "word" if self.word_timestamps else True,
            "generate_kwargs": generate_kwargs,
        }
        with Timer("Whisper MPS transcription"):
            try:
                output = self.pipeline(audio, **inference_kwargs)
            except RuntimeError as exc:
                if batch_size <= 1 or "out of memory" not in str(exc).lower():
                    raise
                logger.warning(
                    "Whisper MPS batch size %s exhausted unified memory; retrying with 1.",
                    batch_size,
                )
                torch.mps.empty_cache()
                inference_kwargs["batch_size"] = 1
                output = self.pipeline(audio, **inference_kwargs)

        if not isinstance(output, dict):
            raise RuntimeError("Whisper MPS returned an unexpected result.")
        return _result_from_pipeline_output(output, language=resolved_language)

    def _warm_up(self) -> None:
        if self.warmup_seconds <= 0 or self.pipeline is None:
            return
        samples = np.zeros(round(self.warmup_seconds * SAMPLE_RATE), dtype=np.float32)
        logger.info("Warming up Whisper on Apple MPS...")
        self._transcribe_input(
            {"array": samples, "sampling_rate": SAMPLE_RATE},
            language=self.language,
            batch_size=1,
        )

    def _require_mps(self) -> None:
        if self.device not in {"auto", "mps"}:
            raise ValueError("Whisper MPS requires the mps or auto device.")
        if not torch.backends.mps.is_built():
            raise RuntimeError("This PyTorch installation was not built with MPS support.")
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "Apple MPS is unavailable. Choose Auto to use Faster-Whisper on CPU instead."
            )

    def _torch_dtype(self) -> torch.dtype:
        if self.dtype_name == "float16":
            return torch.float16
        if self.dtype_name == "float32":
            return torch.float32
        raise ValueError("Whisper MPS dtype must be float16 or float32.")

    def _cache_dir(self) -> Path:
        return model_cache_root(self.config)

    def _setup_cache_environment(self) -> None:
        self.cache_dir = setup_model_cache_environment(self.config)

    def cleanup_models(self) -> None:
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


def _result_from_pipeline_output(
    output: dict[str, Any],
    *,
    language: Optional[str],
) -> TranscriptionResult:
    words: list[WordSegment] = []
    for chunk in output.get("chunks") or []:
        if not isinstance(chunk, dict):
            continue
        text = str(chunk.get("text") or "").strip()
        timestamp = chunk.get("timestamp")
        if not text or not isinstance(timestamp, (tuple, list)) or len(timestamp) != 2:
            continue
        start = _finite_time(timestamp[0])
        end = _finite_time(timestamp[1])
        if start is None:
            continue
        if end is None:
            end = start + 0.01
        end = max(start + 0.001, end)
        words.append(WordSegment(word=text, start=start, end=end))

    segments = _group_words(words)
    text = str(output.get("text") or "").strip()
    if not segments and text:
        segments = [TranscriptSegment(text=text, start=0.0, end=0.0)]
    return TranscriptionResult(
        segments=segments,
        language=language,
        raw={
            "text": text,
            "language": language,
            "segments": [segment.to_dict() for segment in segments],
        },
    )


def _group_words(words: list[WordSegment]) -> list[TranscriptSegment]:
    segments: list[TranscriptSegment] = []
    current: list[WordSegment] = []
    for word in words:
        if current and word.start is not None and current[-1].end is not None:
            if word.start - current[-1].end >= 1.0:
                segments.append(_segment_from_words(current))
                current = []
        current.append(word)
        duration = (word.end or 0.0) - (current[0].start or 0.0)
        if _SENTENCE_END.search(word.word) or duration >= 15.0:
            segments.append(_segment_from_words(current))
            current = []
    if current:
        segments.append(_segment_from_words(current))
    return segments


def _segment_from_words(words: list[WordSegment]) -> TranscriptSegment:
    return TranscriptSegment(
        text=_join_word_text(words),
        start=words[0].start or 0.0,
        end=words[-1].end or words[-1].start or 0.0,
        words=words,
    )


def _join_word_text(words: list[WordSegment]) -> str:
    text = ""
    for word in words:
        token = word.word.strip()
        if not text or token.startswith((".", ",", "!", "?", ":", ";", "\u3002", "\uff01", "\uff1f")):
            text += token
        else:
            text += f" {token}"
    return text


def _finite_time(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number < 0:
        return None
    return number


def _rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


def _device_type(device: Any) -> str:
    return str(getattr(device, "type", device)).split(":", 1)[0]
