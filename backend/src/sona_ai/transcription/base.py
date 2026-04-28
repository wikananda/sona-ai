from typing import Optional, Protocol

from .schemas import TranscriptionResult


class Transcriber(Protocol):
    def load_models(self) -> None:
        ...

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        ...

    def cleanup_models(self) -> None:
        ...

