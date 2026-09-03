import os
from dataclasses import dataclass
from typing import Any, Optional

from sona_ai.services.buffered_live_gateway import BufferedLiveGateway
from sona_ai.services.whisper_live_gateway import WHISPER_LIVE_MODELS


@dataclass(frozen=True)
class WhisperMpsLiveConfig:
    max_sessions: int = 1
    chunk_seconds: float = 2.0
    left_context_seconds: float = 12.0
    right_context_seconds: float = 2.0
    max_session_seconds: float = 6 * 60 * 60
    max_frame_bytes: int = 256 * 1024

    @classmethod
    def from_env(cls) -> "WhisperMpsLiveConfig":
        return cls(
            max_sessions=_positive_int(
                "SONA_WHISPER_MPS_LIVE_MAX_SESSIONS",
                cls.max_sessions,
            ),
            chunk_seconds=_positive_float(
                "SONA_WHISPER_MPS_LIVE_CHUNK_SECONDS",
                cls.chunk_seconds,
            ),
            left_context_seconds=_positive_float(
                "SONA_WHISPER_MPS_LIVE_LEFT_CONTEXT_SECONDS",
                cls.left_context_seconds,
            ),
            right_context_seconds=_positive_float(
                "SONA_WHISPER_MPS_LIVE_RIGHT_CONTEXT_SECONDS",
                cls.right_context_seconds,
            ),
            max_session_seconds=_positive_float(
                "SONA_WHISPER_MPS_LIVE_MAX_SESSION_SECONDS",
                cls.max_session_seconds,
            ),
            max_frame_bytes=_positive_int(
                "SONA_WHISPER_MPS_LIVE_MAX_FRAME_BYTES",
                cls.max_frame_bytes,
            ),
        )


class WhisperMpsLiveGateway(BufferedLiveGateway):
    """Run rolling Whisper inference locally on the Apple GPU."""

    def __init__(
        self,
        transcription_service: Any,
        config: Optional[WhisperMpsLiveConfig] = None,
    ):
        super().__init__(
            transcription_service,
            config or WhisperMpsLiveConfig.from_env(),
            supported_models=set(WHISPER_LIVE_MODELS),
            engine="whisper-mps-live",
            display_name="Whisper MPS",
        )


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
