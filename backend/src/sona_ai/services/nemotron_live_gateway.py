import asyncio
import json
import os
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from sona_ai.core import setup_logging
from sona_ai.services.whisper_live_gateway import (
    WhisperLiveCapacityError,
    WhisperLiveInputError,
    WhisperLiveUnavailableError,
)
from sona_ai.transcription.nemotron_languages import resolve_nemotron_language
from sona_ai.transcription.nemotron_live_protocol import (
    NemotronLiveProtocolError,
    NemotronLiveTranscriptAccumulator,
)
from sona_ai.transcription.whisper_live_protocol import PROTOCOL_VERSION


logger = setup_logging()
ConnectCallable = Callable[..., Any]


@dataclass(frozen=True)
class NemotronLiveConfig:
    url: str = "ws://127.0.0.1:8080/v1/realtime"
    api_key: str = ""
    max_sessions: int = 1
    connect_timeout_seconds: float = 10.0
    ready_timeout_seconds: float = 30.0
    stop_timeout_seconds: float = 60.0
    max_session_seconds: float = 6 * 60 * 60
    max_frame_bytes: int = 256 * 1024
    max_message_bytes: int = 2 * 1024 * 1024

    @classmethod
    def from_env(cls) -> "NemotronLiveConfig":
        return cls(
            url=os.getenv("SONA_NEMOTRON_LIVE_URL", cls.url),
            api_key=os.getenv("SONA_NEMOTRON_API_KEY", cls.api_key),
            max_sessions=_positive_int("SONA_NEMOTRON_LIVE_MAX_SESSIONS", cls.max_sessions),
            connect_timeout_seconds=_positive_float(
                "SONA_NEMOTRON_CONNECT_TIMEOUT", cls.connect_timeout_seconds
            ),
            ready_timeout_seconds=_positive_float(
                "SONA_NEMOTRON_LIVE_READY_TIMEOUT", cls.ready_timeout_seconds
            ),
            stop_timeout_seconds=_positive_float(
                "SONA_NEMOTRON_LIVE_STOP_TIMEOUT", cls.stop_timeout_seconds
            ),
            max_session_seconds=_positive_float(
                "SONA_NEMOTRON_LIVE_MAX_SESSION_SECONDS", cls.max_session_seconds
            ),
            max_frame_bytes=_positive_int(
                "SONA_NEMOTRON_LIVE_MAX_FRAME_BYTES", cls.max_frame_bytes
            ),
            max_message_bytes=_positive_int(
                "SONA_NEMOTRON_LIVE_MAX_MESSAGE_BYTES", cls.max_message_bytes
            ),
        )


class NemotronLiveGateway:
    """Relay browser PCM16 to the NeMo-Speech.cpp realtime endpoint."""

    def __init__(
        self,
        config: Optional[NemotronLiveConfig] = None,
        connect: Optional[ConnectCallable] = None,
    ):
        self.config = config or NemotronLiveConfig.from_env()
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
        if model != "nemotron-3.5":
            raise WhisperLiveInputError("This realtime gateway requires Nemotron 3.5.")
        try:
            resolved_language = resolve_nemotron_language(language)
        except ValueError as exc:
            raise WhisperLiveInputError(str(exc)) from exc

        try:
            await asyncio.wait_for(self._admission.acquire(), timeout=0.05)
        except asyncio.TimeoutError as exc:
            raise WhisperLiveCapacityError(
                "The realtime Nemotron session is busy. Try again shortly."
            ) from exc

        current_task = asyncio.current_task()
        if current_task is not None:
            self._active_relays.add(current_task)
        try:
            await self._relay_admitted(
                browser,
                model=model,
                language=resolved_language,
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
        language: str,
    ) -> None:
        session_id = str(uuid.uuid4())
        accumulator = NemotronLiveTranscriptAccumulator(session_id, language)
        try:
            connection = self._connect(
                _url_with_api_key(self.config.url, self.config.api_key),
                open_timeout=self.config.connect_timeout_seconds,
                max_size=self.config.max_message_bytes,
            )
            async with connection as upstream:
                await self._wait_for_event(upstream, "session.created")
                await upstream.send(json.dumps({
                    "type": "session.update",
                    "session": {
                        "sample_rate": 16000,
                        "language": language,
                        "automatic_punctuation": True,
                        "word_timestamps": True,
                    },
                }))
                await self._wait_for_event(upstream, "session.updated")
                await browser.send_json({
                    "type": "ready",
                    "version": PROTOCOL_VERSION,
                    "session_id": session_id,
                    "engine": "nemotron-live",
                    "model": model,
                    "sample_rate": 16000,
                    "format": "pcm_s16le",
                })
                await self._pump(browser, upstream, accumulator)
        except (WhisperLiveCapacityError, WhisperLiveInputError, WhisperLiveUnavailableError):
            raise
        except NemotronLiveProtocolError as exc:
            raise WhisperLiveUnavailableError(
                "Nemotron returned an invalid realtime response. The recording can still be saved."
            ) from exc
        except asyncio.CancelledError:
            raise
        except (OSError, asyncio.TimeoutError) as exc:
            raise WhisperLiveUnavailableError(
                "Realtime Nemotron is unavailable. The recording can still be saved."
            ) from exc
        except Exception as exc:
            # Connection errors may include their URL; do not risk logging the
            # optional WebSocket api_key query parameter.
            logger.warning(
                "Nemotron realtime relay failed (%s)",
                type(exc).__name__,
            )
            raise WhisperLiveUnavailableError(
                "Realtime Nemotron disconnected. The recording can still be saved."
            ) from exc

    async def _wait_for_event(self, upstream: Any, expected_type: str) -> dict[str, Any]:
        deadline = asyncio.get_running_loop().time() + self.config.ready_timeout_seconds
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise WhisperLiveUnavailableError("Timed out while Nemotron loaded the model.")
            raw = await asyncio.wait_for(upstream.recv(), timeout=remaining)
            message = _decode_message(raw)
            _raise_upstream_error(message)
            if message.get("type") == expected_type:
                return message

    async def _pump(
        self,
        browser: Any,
        upstream: Any,
        accumulator: NemotronLiveTranscriptAccumulator,
    ) -> None:
        stop_requested = asyncio.Event()
        browser_task = asyncio.create_task(
            self._browser_to_upstream(browser, upstream, accumulator, stop_requested),
            name="sona-live-browser-to-nemotron",
        )
        upstream_task = asyncio.create_task(
            self._upstream_to_browser(browser, upstream, accumulator, stop_requested),
            name="sona-live-nemotron-to-browser",
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
                    "Nemotron disconnected before the recording stopped."
                )

            if browser_result == "disconnect":
                upstream_task.cancel()
                return
            try:
                await asyncio.wait_for(upstream_task, timeout=self.config.stop_timeout_seconds)
            except asyncio.TimeoutError:
                upstream_task.cancel()

            await browser.send_json(accumulator.finalize())
        finally:
            for task in (browser_task, upstream_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(browser_task, upstream_task, return_exceptions=True)

    async def _browser_to_upstream(
        self,
        browser: Any,
        upstream: Any,
        accumulator: NemotronLiveTranscriptAccumulator,
        stop_requested: asyncio.Event,
    ) -> str:
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
                accumulator.set_audio_end(audio_bytes / (16000 * 2))
                await upstream.send(frame)
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
                stop_requested.set()
                await upstream.send(json.dumps({"type": "input_audio_buffer.commit"}))
                return "stop"
            raise WhisperLiveInputError("Unsupported live transcription command.")

    async def _upstream_to_browser(
        self,
        browser: Any,
        upstream: Any,
        accumulator: NemotronLiveTranscriptAccumulator,
        stop_requested: asyncio.Event,
    ) -> None:
        commit_acknowledged = False
        completion_after_stop = False
        async for raw in upstream:
            message = _decode_message(raw)
            _raise_upstream_error(message)
            event_type = message.get("type")
            event = None
            if event_type == "conversation.item.input_audio_transcription.delta":
                event = accumulator.apply_delta(message)
            elif event_type == "conversation.item.input_audio_transcription.completed":
                event = accumulator.apply_completed(message)
                if stop_requested.is_set():
                    completion_after_stop = True
            elif event_type == "input_audio_buffer.committed" and stop_requested.is_set():
                commit_acknowledged = True

            if event is not None:
                await browser.send_json(event)
            if stop_requested.is_set() and commit_acknowledged and completion_after_stop:
                return


def _decode_message(raw: Any) -> dict[str, Any]:
    if isinstance(raw, bytes):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise NemotronLiveProtocolError("Nemotron returned non-JSON data") from exc
    try:
        message = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise NemotronLiveProtocolError("Nemotron returned malformed JSON") from exc
    if not isinstance(message, dict):
        raise NemotronLiveProtocolError("Nemotron response must be an object")
    return message


def _raise_upstream_error(message: dict[str, Any]) -> None:
    if message.get("type") != "error":
        return
    error = message.get("error")
    if isinstance(error, dict):
        code = str(error.get("code") or error.get("type") or "").casefold()
        detail = str(error.get("message") or "")
    else:
        code = str(message.get("code") or "").casefold()
        detail = str(error or message.get("message") or "")
    normalized = f"{code} {detail}".casefold()
    if any(value in normalized for value in ("busy", "capacity", "overload", "session_limit")):
        raise WhisperLiveCapacityError("Nemotron is at capacity. Try again shortly.")
    if "invalid_request" in normalized or "invalid session" in normalized:
        raise WhisperLiveInputError("Nemotron rejected the realtime session settings.")
    raise WhisperLiveUnavailableError(
        "Nemotron could not process the realtime session. The recording can still be saved."
    )


def _url_with_api_key(url: str, api_key: str) -> str:
    if not api_key:
        return url
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query["api_key"] = api_key
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


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
