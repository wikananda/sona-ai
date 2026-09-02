import asyncio
import json
import os
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

import numpy as np

from sona_ai.core import setup_logging
from sona_ai.transcription.whisper_live_protocol import (
    PROTOCOL_VERSION,
    WhisperLiveProtocolError,
    WhisperLiveTranscriptAccumulator,
)


logger = setup_logging()

ConnectCallable = Callable[..., Any]

WHISPER_LIVE_MODELS = {
    "faster-whisper-turbo": "turbo",
    "faster-whisper-large-v3": "large-v3",
}


class WhisperLiveError(RuntimeError):
    code = "LIVE_TRANSCRIPTION_FAILED"


class WhisperLiveUnavailableError(WhisperLiveError):
    code = "LIVE_TRANSCRIPTION_UNAVAILABLE"


class WhisperLiveCapacityError(WhisperLiveError):
    code = "LIVE_TRANSCRIPTION_BUSY"


class WhisperLiveInputError(WhisperLiveError):
    code = "INVALID_LIVE_AUDIO"


@dataclass(frozen=True)
class WhisperLiveConfig:
    url: str = "ws://127.0.0.1:9090"
    max_sessions: int = 1
    connect_timeout_seconds: float = 10.0
    ready_timeout_seconds: float = 120.0
    stop_timeout_seconds: float = 20.0
    max_session_seconds: float = 6 * 60 * 60
    max_frame_bytes: int = 256 * 1024

    @classmethod
    def from_env(cls) -> "WhisperLiveConfig":
        return cls(
            url=os.getenv("SONA_WHISPER_LIVE_URL", cls.url),
            max_sessions=_positive_int("SONA_WHISPER_LIVE_MAX_SESSIONS", cls.max_sessions),
            connect_timeout_seconds=_positive_float(
                "SONA_WHISPER_LIVE_CONNECT_TIMEOUT", cls.connect_timeout_seconds
            ),
            ready_timeout_seconds=_positive_float(
                "SONA_WHISPER_LIVE_READY_TIMEOUT", cls.ready_timeout_seconds
            ),
            stop_timeout_seconds=_positive_float(
                "SONA_WHISPER_LIVE_STOP_TIMEOUT", cls.stop_timeout_seconds
            ),
            max_session_seconds=_positive_float(
                "SONA_WHISPER_LIVE_MAX_SESSION_SECONDS", cls.max_session_seconds
            ),
            max_frame_bytes=_positive_int(
                "SONA_WHISPER_LIVE_MAX_FRAME_BYTES", cls.max_frame_bytes
            ),
        )


class WhisperLiveGateway:
    """Relay browser PCM to WhisperLive while hiding its wire protocol."""

    def __init__(
        self,
        config: Optional[WhisperLiveConfig] = None,
        connect: Optional[ConnectCallable] = None,
    ):
        self.config = config or WhisperLiveConfig.from_env()
        self._connect = connect or _websocket_connect
        self._admission = asyncio.Semaphore(self.config.max_sessions)
        self._active_relays: set[asyncio.Task] = set()

    async def relay(
        self,
        browser: Any,
        *,
        model: str,
        language: Optional[str],
    ) -> None:
        upstream_model = WHISPER_LIVE_MODELS.get(model)
        if upstream_model is None:
            raise WhisperLiveInputError("Realtime streaming is available only for Whisper models.")

        try:
            await asyncio.wait_for(self._admission.acquire(), timeout=0.05)
        except asyncio.TimeoutError as exc:
            raise WhisperLiveCapacityError("All realtime transcription sessions are busy.") from exc

        current_task = asyncio.current_task()
        if current_task is not None:
            self._active_relays.add(current_task)
        try:
            await self._relay_admitted(
                browser,
                model=upstream_model,
                language=language,
            )
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

    async def _relay_admitted(
        self,
        browser: Any,
        *,
        model: str,
        language: Optional[str],
    ) -> None:
        session_id = str(uuid.uuid4())
        accumulator = WhisperLiveTranscriptAccumulator(session_id)
        if language:
            accumulator.set_language(language)

        try:
            connection = self._connect(
                self.config.url,
                open_timeout=self.config.connect_timeout_seconds,
                max_size=2 * 1024 * 1024,
            )
            async with connection as upstream:
                await upstream.send(json.dumps(_upstream_options(
                    session_id=session_id,
                    model=model,
                    language=language,
                )))
                await self._wait_until_ready(upstream, accumulator)
                await browser.send_json({
                    "type": "ready",
                    "version": PROTOCOL_VERSION,
                    "session_id": session_id,
                    "engine": "whisper-live",
                    "model": model,
                    "sample_rate": 16000,
                    "format": "pcm_s16le",
                })
                await self._pump(browser, upstream, accumulator)
        except (WhisperLiveError, WhisperLiveProtocolError):
            raise
        except (OSError, asyncio.TimeoutError) as exc:
            raise WhisperLiveUnavailableError(
                "Realtime Whisper is unavailable. The recording can still be saved."
            ) from exc
        except Exception as exc:
            logger.warning("WhisperLive relay failed: %s", exc)
            raise WhisperLiveUnavailableError(
                "Realtime Whisper disconnected. The recording can still be saved."
            ) from exc

    async def _wait_until_ready(
        self,
        upstream: Any,
        accumulator: WhisperLiveTranscriptAccumulator,
    ) -> None:
        deadline = asyncio.get_running_loop().time() + self.config.ready_timeout_seconds
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise WhisperLiveUnavailableError("Timed out while WhisperLive loaded the model.")
            raw = await asyncio.wait_for(upstream.recv(), timeout=remaining)
            message = _decode_upstream_message(raw)
            _validate_upstream_status(message)
            if message.get("language"):
                accumulator.set_language(message["language"])
            if message.get("message") == "SERVER_READY":
                return

    async def _pump(
        self,
        browser: Any,
        upstream: Any,
        accumulator: WhisperLiveTranscriptAccumulator,
    ) -> None:
        browser_task = asyncio.create_task(
            self._browser_to_upstream(browser, upstream),
            name="sona-live-browser-to-whisper",
        )
        upstream_task = asyncio.create_task(
            self._upstream_to_browser(browser, upstream, accumulator),
            name="sona-live-whisper-to-browser",
        )
        try:
            done, _ = await asyncio.wait(
                {browser_task, upstream_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if browser_task in done:
                browser_result = await browser_task
            else:
                await upstream_task
                browser_task.cancel()
                raise WhisperLiveUnavailableError(
                    "WhisperLive disconnected before the recording stopped."
                )

            if browser_result == "stop":
                try:
                    await asyncio.wait_for(
                        upstream_task,
                        timeout=self.config.stop_timeout_seconds,
                    )
                except asyncio.TimeoutError:
                    upstream_task.cancel()
            else:
                upstream_task.cancel()
                return

            await browser.send_json(accumulator.finalize())
        finally:
            for task in (browser_task, upstream_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(browser_task, upstream_task, return_exceptions=True)

    async def _browser_to_upstream(self, browser: Any, upstream: Any) -> str:
        audio_bytes = 0
        max_audio_bytes = int(self.config.max_session_seconds * 16000 * 2)
        while True:
            message = await browser.receive()
            if message.get("type") == "websocket.disconnect":
                return "disconnect"

            frame = message.get("bytes")
            if frame is not None:
                if not frame:
                    continue
                if len(frame) > self.config.max_frame_bytes:
                    raise WhisperLiveInputError("Live audio frame is too large.")
                if len(frame) % 2:
                    raise WhisperLiveInputError("Live audio must contain complete int16 samples.")
                audio_bytes += len(frame)
                if audio_bytes > max_audio_bytes:
                    raise WhisperLiveInputError("Live transcription reached its duration limit.")
                await upstream.send(_pcm16_to_float32(frame))
                continue

            text = message.get("text")
            if text is None:
                continue
            try:
                command = json.loads(text)
            except json.JSONDecodeError as exc:
                raise WhisperLiveInputError("Live transcription command must be valid JSON.") from exc
            if isinstance(command, dict) and command.get("type") == "stop":
                await upstream.send(b"END_OF_AUDIO")
                return "stop"
            raise WhisperLiveInputError("Unsupported live transcription command.")

    async def _upstream_to_browser(
        self,
        browser: Any,
        upstream: Any,
        accumulator: WhisperLiveTranscriptAccumulator,
    ) -> None:
        async for raw in upstream:
            message = _decode_upstream_message(raw)
            if message.get("uid") not in {None, accumulator.session_id}:
                continue
            _validate_upstream_status(message)
            if message.get("message") == "DISCONNECT":
                raise WhisperLiveUnavailableError("WhisperLive ended the session.")
            if "language" in message:
                event = accumulator.set_language(message.get("language"))
                if event is not None:
                    await browser.send_json(event)
            if "segments" in message:
                event = accumulator.apply_segments(message["segments"])
                if event is not None:
                    await browser.send_json(event)


def _upstream_options(
    *,
    session_id: str,
    model: str,
    language: Optional[str],
) -> dict[str, Any]:
    return {
        "uid": session_id,
        "language": language,
        "task": "transcribe",
        "model": model,
        "use_vad": True,
        "send_last_n_segments": 10,
        "no_speech_thresh": 0.45,
        "clip_audio": True,
        "same_output_threshold": 7,
        "word_timestamps": True,
        "enable_diarization": False,
    }


def _decode_upstream_message(raw: Any) -> dict[str, Any]:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        message = json.loads(raw)
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WhisperLiveProtocolError("WhisperLive returned malformed JSON") from exc
    if not isinstance(message, dict):
        raise WhisperLiveProtocolError("WhisperLive response must be an object")
    return message


def _validate_upstream_status(message: dict[str, Any]) -> None:
    status = message.get("status")
    if status == "WAIT":
        raise WhisperLiveCapacityError("WhisperLive is at capacity. Try again shortly.")
    if status == "ERROR":
        raise WhisperLiveUnavailableError("WhisperLive could not start this session.")


def _pcm16_to_float32(frame: bytes) -> bytes:
    samples = np.frombuffer(frame, dtype="<i2")
    return (samples.astype(np.float32) / 32768.0).tobytes()


def _websocket_connect(url: str, **kwargs: Any) -> Any:
    try:
        from websockets.asyncio.client import connect
    except ImportError:
        from websockets import connect
    return connect(url, **kwargs)


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
