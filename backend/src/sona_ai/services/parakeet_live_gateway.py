import asyncio
import json
import os
import uuid
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from sona_ai.core import setup_logging
from sona_ai.services.whisper_live_gateway import (
    WhisperLiveCapacityError,
    WhisperLiveInputError,
    WhisperLiveUnavailableError,
)
from sona_ai.transcription.parakeet_live_protocol import (
    ParakeetLiveProtocolError,
    ParakeetLiveTranscriptAccumulator,
)
from sona_ai.transcription.whisper_live_protocol import PROTOCOL_VERSION


logger = setup_logging()
SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2


@dataclass(frozen=True)
class ParakeetLiveConfig:
    max_sessions: int = 1
    chunk_seconds: float = 2.0
    left_context_seconds: float = 10.0
    right_context_seconds: float = 2.0
    max_session_seconds: float = 6 * 60 * 60
    max_frame_bytes: int = 256 * 1024

    @classmethod
    def from_env(cls) -> "ParakeetLiveConfig":
        return cls(
            max_sessions=_positive_int(
                "SONA_PARAKEET_LIVE_MAX_SESSIONS",
                cls.max_sessions,
            ),
            chunk_seconds=_positive_float(
                "SONA_PARAKEET_LIVE_CHUNK_SECONDS",
                cls.chunk_seconds,
            ),
            left_context_seconds=_positive_float(
                "SONA_PARAKEET_LIVE_LEFT_CONTEXT_SECONDS",
                cls.left_context_seconds,
            ),
            right_context_seconds=_positive_float(
                "SONA_PARAKEET_LIVE_RIGHT_CONTEXT_SECONDS",
                cls.right_context_seconds,
            ),
            max_session_seconds=_positive_float(
                "SONA_PARAKEET_LIVE_MAX_SESSION_SECONDS",
                cls.max_session_seconds,
            ),
            max_frame_bytes=_positive_int(
                "SONA_PARAKEET_LIVE_MAX_FRAME_BYTES",
                cls.max_frame_bytes,
            ),
        )


class ParakeetLiveGateway:
    """Run low-latency Parakeet decoding over a bounded rolling PCM window."""

    def __init__(self, transcription_service: Any, config: Optional[ParakeetLiveConfig] = None):
        self.transcription_service = transcription_service
        self.config = config or ParakeetLiveConfig.from_env()
        self._admission = asyncio.Semaphore(self.config.max_sessions)
        self._active_relays: set[asyncio.Task] = set()

    async def relay(
        self,
        browser: Any,
        *,
        model: str,
        device: str,
        language: Optional[str],
    ) -> None:
        if model != "parakeet":
            raise WhisperLiveInputError("This realtime gateway requires the Parakeet model.")

        try:
            await asyncio.wait_for(self._admission.acquire(), timeout=0.05)
        except asyncio.TimeoutError as exc:
            raise WhisperLiveCapacityError(
                "The realtime Parakeet session is busy. Try again shortly."
            ) from exc

        current_task = asyncio.current_task()
        if current_task is not None:
            self._active_relays.add(current_task)
        try:
            await asyncio.to_thread(
                self.transcription_service.prepare_live_transcription,
                model=model,
                device=device,
            )
            session_id = str(uuid.uuid4())
            await browser.send_json({
                "type": "ready",
                "version": PROTOCOL_VERSION,
                "session_id": session_id,
                "engine": "parakeet-live",
                "model": model,
                "sample_rate": SAMPLE_RATE,
                "format": "pcm_s16le",
            })
            relay = _ParakeetRelay(
                browser=browser,
                transcription_service=self.transcription_service,
                config=self.config,
                session_id=session_id,
                model=model,
                device=device,
                language=language,
            )
            await relay.run()
        except (WhisperLiveCapacityError, WhisperLiveInputError):
            raise
        except (ParakeetLiveProtocolError, ValueError) as exc:
            raise WhisperLiveInputError(str(exc)) from exc
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Realtime Parakeet relay failed: %s", exc)
            raise WhisperLiveUnavailableError(
                "Realtime Parakeet is unavailable. The recording can still be saved."
            ) from exc
        finally:
            if current_task is not None:
                self._active_relays.discard(current_task)
            self._admission.release()

    async def close(self) -> None:
        current = asyncio.current_task()
        tasks = [task for task in self._active_relays if task is not current]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


class _ParakeetRelay:
    def __init__(
        self,
        *,
        browser: Any,
        transcription_service: Any,
        config: ParakeetLiveConfig,
        session_id: str,
        model: str,
        device: str,
        language: Optional[str],
    ):
        self.browser = browser
        self.transcription_service = transcription_service
        self.config = config
        self.model = model
        self.device = device
        self.language = language
        self.accumulator = ParakeetLiveTranscriptAccumulator(session_id, language)
        self.audio = bytearray()
        self.total_samples = 0
        self.last_decoded_samples = 0
        self.stopping = False
        self.disconnected = False
        self.audio_changed = asyncio.Event()

        self.chunk_samples = round(config.chunk_seconds * SAMPLE_RATE)
        self.right_context_samples = round(config.right_context_seconds * SAMPLE_RATE)
        self.left_context_samples = round(config.left_context_seconds * SAMPLE_RATE)
        self.first_decode_samples = self.chunk_samples + self.right_context_samples
        self.window_samples = (
            self.left_context_samples
            + self.chunk_samples
            + self.right_context_samples
        )
        self.max_session_samples = round(config.max_session_seconds * SAMPLE_RATE)

    async def run(self) -> None:
        reader = asyncio.create_task(self._read_browser(), name="sona-parakeet-live-reader")
        decoder = asyncio.create_task(self._decode_audio(), name="sona-parakeet-live-decoder")
        tasks = {reader, decoder}
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
            for task in done:
                exception = task.exception()
                if exception is not None:
                    raise exception
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _read_browser(self) -> None:
        while True:
            message = await self.browser.receive()
            if message.get("type") == "websocket.disconnect":
                self.disconnected = True
                self.audio_changed.set()
                return

            frame = message.get("bytes")
            if frame is not None:
                self._append_audio(frame)
                self.audio_changed.set()
                continue

            text = message.get("text")
            if text is None:
                continue
            try:
                command = json.loads(text)
            except json.JSONDecodeError as exc:
                raise WhisperLiveInputError(
                    "Live transcription command must be valid JSON."
                ) from exc
            if isinstance(command, dict) and command.get("type") == "stop":
                self.stopping = True
                self.audio_changed.set()
                return
            raise WhisperLiveInputError("Unsupported live transcription command.")

    def _append_audio(self, frame: bytes) -> None:
        if not frame:
            return
        if len(frame) > self.config.max_frame_bytes:
            raise WhisperLiveInputError("Live audio frame is too large.")
        if len(frame) % BYTES_PER_SAMPLE:
            raise WhisperLiveInputError("Live audio must contain complete int16 samples.")

        frame_samples = len(frame) // BYTES_PER_SAMPLE
        if self.total_samples + frame_samples > self.max_session_samples:
            raise WhisperLiveInputError("Live transcription reached its duration limit.")
        self.total_samples += frame_samples
        self.audio.extend(frame)

        max_bytes = self.window_samples * BYTES_PER_SAMPLE
        overflow = len(self.audio) - max_bytes
        if overflow > 0:
            del self.audio[:overflow]

    async def _decode_audio(self) -> None:
        while True:
            await self.audio_changed.wait()
            self.audio_changed.clear()

            while True:
                if self.disconnected:
                    return
                final = self.stopping
                if not final and not self._normal_decode_due():
                    break
                if final and self.total_samples == 0:
                    await self.browser.send_json(self.accumulator.finalize())
                    return

                snapshot_end = self.total_samples
                samples, window_start = self._snapshot(final=final)
                if window_start > self.accumulator.commit_horizon + 0.05:
                    raise WhisperLiveUnavailableError(
                        "Realtime Parakeet could not keep up. The recording can still be saved."
                    )

                transcription = await asyncio.to_thread(
                    self.transcription_service.transcribe_live_samples,
                    samples,
                    language=self.language,
                    model=self.model,
                    device=self.device,
                )
                if self.disconnected:
                    return

                audio_end = snapshot_end / SAMPLE_RATE
                stable_cutoff = (
                    audio_end
                    if final
                    else max(0.0, audio_end - self.config.right_context_seconds)
                )
                event = self.accumulator.apply_snapshot(
                    transcription.to_segment_dicts(),
                    window_start=window_start,
                    stable_cutoff=stable_cutoff,
                    audio_end=audio_end,
                )
                self.last_decoded_samples = snapshot_end
                if event is not None:
                    await self.browser.send_json(event)

                if final:
                    await self.browser.send_json(self.accumulator.finalize())
                    return

    def _normal_decode_due(self) -> bool:
        return (
            self.total_samples >= self.first_decode_samples
            and self.total_samples - self.last_decoded_samples >= self.chunk_samples
        )

    def _snapshot(self, *, final: bool) -> tuple[np.ndarray, float]:
        pcm = bytes(self.audio)
        available_samples = len(pcm) // BYTES_PER_SAMPLE
        if final:
            max_actual_samples = self.left_context_samples + self.chunk_samples
            if available_samples > max_actual_samples:
                pcm = pcm[-max_actual_samples * BYTES_PER_SAMPLE :]
                available_samples = max_actual_samples

        window_start_samples = self.total_samples - available_samples
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        if final and self.right_context_samples:
            samples = np.concatenate((
                samples,
                np.zeros(self.right_context_samples, dtype=np.float32),
            ))
        return samples, window_start_samples / SAMPLE_RATE


def _positive_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


def _positive_float(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default
