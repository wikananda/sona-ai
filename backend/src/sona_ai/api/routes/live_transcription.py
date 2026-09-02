import asyncio
import json
from typing import Any, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from sona_ai.core import setup_logging, validate_device_available
from sona_ai.db.engine import SessionLocal
from sona_ai.db.models import Project
from sona_ai.services.whisper_live_gateway import (
    WHISPER_LIVE_MODELS,
    WhisperLiveCapacityError,
    WhisperLiveError,
    WhisperLiveInputError,
    WhisperLiveUnavailableError,
)


logger = setup_logging()
router = APIRouter()
START_TIMEOUT_SECONDS = 15.0
MAX_START_MESSAGE_BYTES = 16 * 1024


@router.websocket("/projects/{project_id}/live-transcription/ws")
async def live_transcription_socket(websocket: WebSocket, project_id: str) -> None:
    await websocket.accept()
    close_code = 1000
    try:
        start = await _receive_start(websocket)
        _require_project(project_id)
        gateway = getattr(websocket.app.state, "whisper_live_gateway", None)
        if gateway is None:
            raise WhisperLiveUnavailableError("Realtime Whisper is not configured.")
        await gateway.relay(
            websocket,
            model=start["model"],
            language=start["language"],
        )
    except WebSocketDisconnect:
        return
    except WhisperLiveInputError as exc:
        close_code = 1008
        await _send_error(websocket, exc, recoverable=False)
    except WhisperLiveCapacityError as exc:
        close_code = 1013
        await _send_error(websocket, exc, recoverable=True)
    except WhisperLiveUnavailableError as exc:
        close_code = 1013
        await _send_error(websocket, exc, recoverable=True)
    except WhisperLiveError as exc:
        close_code = 1011
        await _send_error(websocket, exc, recoverable=True)
    except Exception as exc:
        close_code = 1011
        logger.exception("Unexpected live transcription failure")
        await _send_error(
            websocket,
            WhisperLiveError("Realtime transcription failed. The recording can still be saved."),
            recoverable=True,
        )
    finally:
        try:
            await websocket.close(code=close_code)
        except (RuntimeError, WebSocketDisconnect):
            pass


async def _receive_start(websocket: WebSocket) -> dict[str, Any]:
    try:
        message = await asyncio.wait_for(
            websocket.receive(),
            timeout=START_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError as exc:
        raise WhisperLiveInputError("Timed out waiting for live transcription settings.") from exc

    if message.get("type") == "websocket.disconnect":
        raise WebSocketDisconnect(message.get("code", 1000))
    raw = message.get("text")
    if raw is None:
        raise WhisperLiveInputError("Send live transcription settings before audio.")
    if len(raw.encode("utf-8")) > MAX_START_MESSAGE_BYTES:
        raise WhisperLiveInputError("Live transcription settings are too large.")
    try:
        body = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise WhisperLiveInputError("Live transcription settings must be valid JSON.") from exc
    return _validate_start(body)


def _validate_start(body: Any) -> dict[str, Any]:
    if not isinstance(body, dict) or body.get("type") != "start":
        raise WhisperLiveInputError("The first message must be a live transcription start request.")
    if body.get("version") != 1:
        raise WhisperLiveInputError("Unsupported live transcription protocol version.")

    model = str(body.get("model") or "").lower().strip()
    if model not in WHISPER_LIVE_MODELS:
        raise WhisperLiveInputError("Realtime streaming is available only for Whisper models.")

    device = str(body.get("device") or "auto").lower().strip()
    try:
        validate_device_available(device)
    except ValueError as exc:
        raise WhisperLiveInputError(str(exc)) from exc

    language = _normalize_language(body.get("language"))
    audio = body.get("audio")
    if not isinstance(audio, dict):
        raise WhisperLiveInputError("Live audio settings are required.")
    if audio.get("encoding") != "pcm_s16le":
        raise WhisperLiveInputError("Live audio must use pcm_s16le encoding.")
    if audio.get("sample_rate") != 16000 or audio.get("channels") != 1:
        raise WhisperLiveInputError("Live audio must be mono at 16 kHz.")

    return {
        "model": model,
        "device": device,
        "language": language,
    }


def _normalize_language(language: Any) -> Optional[str]:
    if language is None:
        return None
    normalized = str(language).strip()
    if not normalized or normalized.lower() in {"auto", "none"}:
        return None
    if len(normalized) > 32:
        raise WhisperLiveInputError("Language code is too long.")
    return normalized


def _require_project(project_id: str) -> None:
    db = SessionLocal()
    try:
        if db.get(Project, project_id) is None:
            raise WhisperLiveInputError("Project not found.")
    finally:
        db.close()


async def _send_error(
    websocket: WebSocket,
    error: WhisperLiveError,
    *,
    recoverable: bool,
) -> None:
    try:
        await websocket.send_json({
            "type": "error",
            "version": 1,
            "code": error.code,
            "message": str(error),
            "recoverable": recoverable,
        })
    except (RuntimeError, WebSocketDisconnect):
        pass
