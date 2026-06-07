import json
import os
import wave
import subprocess
import tempfile
import time
from pathlib import Path

from sona_ai.core import PROJECT_ROOT, resolve_device, setup_logging
from sona_ai.transcription.schemas import TranscriptionResult


logger = setup_logging()


class ExternalWav2Vec2Aligner:
    def __init__(self, config: dict):
        self.config = config
        alignment_config = config.get("alignment", {})
        self.conda_env = alignment_config.get("conda_env", "sona-aligner")
        self.tool_path = PROJECT_ROOT / alignment_config.get(
            "tool_path",
            "tools/alignment/align_whisperx_wav2vec2.py",
        )
        self.device = resolve_device(config.get("model", {}).get("device", "cpu"), no_mps=True)
        cache_root = PROJECT_ROOT / config.get("cp_dir", {}).get("hf_cache", "cp/hf_cache")
        self.cache_dir = cache_root / "wav2vec2-align"
        self.timeout_seconds = int(alignment_config.get("timeout_seconds", 900))

    def load_models(self) -> None:
        if not self.tool_path.is_file():
            raise FileNotFoundError(f"Wav2Vec2 alignment tool not found: {self.tool_path}")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def align(
        self,
        transcription: TranscriptionResult,
        audio_path: str,
    ) -> TranscriptionResult:
        language = self._resolve_language(transcription.language)
        model_name = self._align_model_name(language)
        raw_segments = transcription.to_segment_dicts()
        input_segments = self._normalize_segments_for_alignment(raw_segments, audio_path)
        logger.info(
            "Preparing external Wav2Vec2 alignment input: segments=%d timed_segments=%d words=%d",
            len(input_segments),
            self._timed_segment_count(input_segments),
            self._word_count(input_segments),
        )

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as input_file:
            input_path = Path(input_file.name)
            json.dump(
                {
                    "language": transcription.language,
                    "segments": input_segments,
                },
                input_file,
            )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as output_file:
            output_path = Path(output_file.name)

        cmd = [
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            self.conda_env,
            "python",
            str(self.tool_path),
            audio_path,
            str(input_path),
            str(output_path),
            "--language",
            language,
            "--device",
            self.device,
            "--model-name",
            model_name,
            "--cache-dir",
            str(self.cache_dir),
        ]

        logger.info("Running external Wav2Vec2 alignment with %s model: %s", language, model_name)
        started_at = time.time()
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        try:
            subprocess.run(
                cmd,
                check=True,
                cwd=PROJECT_ROOT,
                env=env,
                timeout=self.timeout_seconds,
            )
            with output_path.open("r") as f:
                aligned = json.load(f)
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(
                "External Wav2Vec2 alignment timed out after "
                f"{self.timeout_seconds} seconds. The aligner may still be "
                "downloading/loading the model or stuck in the external environment."
            ) from exc
        finally:
            input_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)

        logger.info("External Wav2Vec2 alignment finished in %.2f seconds", time.time() - started_at)
        aligned.setdefault("language", transcription.language or language)
        aligned_result = TranscriptionResult.from_aligned_result(aligned)
        logger.info(
            "External Wav2Vec2 alignment output: segments=%d timed_segments=%d timed_words=%d",
            len(aligned_result.segments),
            self._timed_result_segment_count(aligned_result),
            self._timed_result_word_count(aligned_result),
        )
        return aligned_result

    def _normalize_segments_for_alignment(
        self,
        segments: list[dict],
        audio_path: str,
    ) -> list[dict]:
        if not segments:
            return segments

        if any(self._is_timed_segment(segment) for segment in segments):
            return segments

        duration = self._audio_duration(audio_path)
        if duration <= 0:
            return segments

        normalized_segments = []
        for segment in segments:
            normalized_segment = dict(segment)
            normalized_segment["start"] = 0.0
            normalized_segment["end"] = duration
            normalized_segment.pop("words", None)
            normalized_segments.append(normalized_segment)

        logger.info(
            "Normalized %d untimed alignment segment(s) to full audio duration %.3fs",
            len(normalized_segments),
            duration,
        )
        return normalized_segments

    def _audio_duration(self, audio_path: str) -> float:
        try:
            import soundfile as sf

            info = sf.info(audio_path)
            return float(info.duration or 0.0)
        except Exception:
            pass

        try:
            with wave.open(audio_path) as audio:
                frames = audio.getnframes()
                rate = audio.getframerate()
                if rate > 0:
                    return frames / float(rate)
        except Exception:
            logger.warning("Unable to determine audio duration for alignment: %s", audio_path)

        return 0.0

    def _is_timed_segment(self, segment: dict) -> bool:
        return float(segment.get("end") or 0.0) > float(segment.get("start") or 0.0)

    def _timed_segment_count(self, segments: list[dict]) -> int:
        return sum(1 for segment in segments if self._is_timed_segment(segment))

    def _word_count(self, segments: list[dict]) -> int:
        return sum(len(segment.get("words") or []) for segment in segments)

    def _timed_result_segment_count(self, transcription: TranscriptionResult) -> int:
        return sum(
            1
            for segment in transcription.segments
            if segment.end > segment.start
        )

    def _timed_result_word_count(self, transcription: TranscriptionResult) -> int:
        return sum(
            1
            for segment in transcription.segments
            for word in segment.words
            if word.start is not None and word.end is not None and word.end > word.start
        )

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

    def cleanup_models(self) -> None:
        return None
