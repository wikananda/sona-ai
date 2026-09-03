import math
import os
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any, Callable, Optional

import httpx

from sona_ai.core import Timer, setup_logging
from sona_ai.transcription.nemotron_languages import resolve_nemotron_language
from sona_ai.transcription.schemas import TranscriptSegment, TranscriptionResult, WordSegment


logger = setup_logging()

NEMOTRON_MODEL_REPO = "nvidia/nemotron-3.5-asr-streaming-0.6b"
NEMOTRON_GGUF_FILENAME = "nemotron-3.5-asr-streaming-0.6b.q8_0.gguf"
NEMOTRON_MODEL_REVISION = "1c8deaecc64b91f034d73e08dd8b64625eb3395d"
NEMOTRON_GGUF_SIZE_BYTES = 741_548_352

ClientFactory = Callable[..., httpx.Client]


class NemotronTranscriber:
    """Whole-file transcription through a local NeMo-Speech.cpp server."""

    def __init__(
        self,
        config: dict,
        *,
        client_factory: ClientFactory = httpx.Client,
    ):
        self.config = config
        model_config = config.get("model", {})
        server_config = config.get("server", {})
        self.model_name = model_config.get("model_name", NEMOTRON_MODEL_REPO)
        self.runtime_model = model_config.get("runtime_model", "default")
        self.language = model_config.get("language")
        # Kept for Sona's pipeline cache key. Compute placement is controlled by
        # the isolated NeMo-Speech.cpp process, not this Python client.
        self.device = model_config.get("device", "auto")
        self.server_url = os.getenv(
            "SONA_NEMOTRON_URL",
            server_config.get("url", "http://127.0.0.1:8080"),
        ).rstrip("/")
        self.api_key = os.getenv(
            "SONA_NEMOTRON_API_KEY",
            server_config.get("api_key", ""),
        ).strip()
        self.connect_timeout = _positive_float(
            os.getenv("SONA_NEMOTRON_CONNECT_TIMEOUT"),
            server_config.get("connect_timeout_seconds", 10.0),
        )
        self.request_timeout = _positive_float(
            os.getenv("SONA_NEMOTRON_REQUEST_TIMEOUT"),
            server_config.get("request_timeout_seconds", 600.0),
        )
        self._client_factory = client_factory
        self._client: Optional[httpx.Client] = None

    def load_models(self) -> None:
        if self._client is not None:
            return

        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        client = self._client_factory(
            base_url=self.server_url,
            headers=headers,
            timeout=httpx.Timeout(
                self.request_timeout,
                connect=self.connect_timeout,
            ),
        )
        try:
            response = client.get("/ready")
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict) or not (
                payload.get("ready") is True or payload.get("status") == "ready"
            ):
                raise RuntimeError("Nemotron server is not ready.")
        except Exception:
            client.close()
            raise RuntimeError(
                "Nemotron 3.5 is unavailable. Start the local NeMo-Speech.cpp server."
            ) from None

        self._client = client
        logger.info("Connected to Nemotron 3.5 at %s", self.server_url)

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        if self._client is None:
            raise ReferenceError("Nemotron transcription client is not initialized.")

        resolved_language = resolve_nemotron_language(language or self.language)
        wav_path, temporary = _ensure_pcm16_wav(Path(audio_path))
        try:
            logger.info("Running Nemotron 3.5 whole-file transcription...")
            with Timer("Nemotron 3.5 transcription"):
                with wav_path.open("rb") as audio_file:
                    response = self._client.post(
                        "/v1/audio/transcriptions",
                        data={
                            "model": self.runtime_model,
                            "language": resolved_language,
                            "response_format": "verbose_json",
                        },
                        files={"file": (wav_path.name, audio_file, "audio/wav")},
                    )
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise RuntimeError("Nemotron 3.5 whole-file transcription failed.") from exc
        finally:
            if temporary:
                wav_path.unlink(missing_ok=True)

        return _result_from_payload(payload, fallback_language=resolved_language)

    def cleanup_models(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None


def _ensure_pcm16_wav(audio_path: Path) -> tuple[Path, bool]:
    if _is_pcm16_mono_16khz_wav(audio_path):
        return audio_path, False

    with tempfile.NamedTemporaryFile(prefix="sona-nemotron-", suffix=".wav", delete=False) as output:
        output_path = Path(output.name)
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(audio_path),
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(output_path),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        output_path.unlink(missing_ok=True)
        raise RuntimeError("Could not convert audio for Nemotron 3.5.") from exc
    return output_path, True


def _is_pcm16_mono_16khz_wav(audio_path: Path) -> bool:
    try:
        with wave.open(str(audio_path), "rb") as wav_file:
            return (
                wav_file.getnchannels() == 1
                and wav_file.getsampwidth() == 2
                and wav_file.getframerate() == 16000
                and wav_file.getcomptype() == "NONE"
            )
    except (OSError, EOFError, wave.Error):
        return False


def _result_from_payload(
    payload: Any,
    *,
    fallback_language: Optional[str],
) -> TranscriptionResult:
    if not isinstance(payload, dict):
        raise RuntimeError("Nemotron 3.5 returned an invalid transcription response.")

    text = str(payload.get("text") or payload.get("transcript") or "").strip()
    raw_words = payload.get("words")
    if not isinstance(raw_words, list):
        words_info = payload.get("words_info")
        raw_words = words_info.get("words", []) if isinstance(words_info, dict) else []

    words: list[WordSegment] = []
    previous_start = 0.0
    for raw_word in raw_words:
        if not isinstance(raw_word, dict):
            continue
        word_text = str(raw_word.get("word") or raw_word.get("text") or "").strip()
        if not word_text:
            continue
        start = _finite_nonnegative(raw_word.get("start", raw_word.get("start_time")))
        end = _finite_nonnegative(raw_word.get("end", raw_word.get("end_time")))
        if start is not None:
            start = max(previous_start, start)
            previous_start = start
        if start is not None and end is not None:
            end = max(start, end)
        confidence = _finite_nonnegative(raw_word.get("confidence", raw_word.get("score")))
        if confidence is not None:
            confidence = min(confidence, 1.0)
        speaker = raw_word.get("speaker", raw_word.get("speaker_tag"))
        words.append(WordSegment(
            word=word_text,
            start=start,
            end=end,
            score=confidence,
            speaker=str(speaker) if speaker is not None else None,
        ))

    duration = _finite_nonnegative(payload.get("duration")) or 0.0
    timed_starts = [word.start for word in words if word.start is not None]
    timed_ends = [word.end for word in words if word.end is not None]
    start = min(timed_starts) if timed_starts else 0.0
    end = max([duration, *timed_ends]) if timed_ends else duration
    speakers = {word.speaker for word in words if word.speaker}
    segment_speaker = speakers.pop() if len(speakers) == 1 else None
    segment = TranscriptSegment(
        text=text,
        start=start,
        end=max(start, end),
        speaker=segment_speaker,
        words=words,
    )
    detected_language = payload.get("language") or fallback_language
    return TranscriptionResult(
        segments=[segment],
        language=str(detected_language) if detected_language else None,
        raw=payload,
    )


def _finite_nonnegative(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return max(0.0, number)


def _positive_float(value: Any, default: Any) -> float:
    try:
        number = float(value if value is not None else default)
    except (TypeError, ValueError):
        return float(default)
    return number if number > 0 else float(default)
