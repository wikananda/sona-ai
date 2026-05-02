from typing import Protocol

from sona_ai.transcription.schemas import TranscriptionResult


class Aligner(Protocol):
    def load_models(self) -> None:
        ...

    def align(
        self,
        transcription: TranscriptionResult,
        audio_path: str,
    ) -> TranscriptionResult:
        ...

    def cleanup_models(self) -> None:
        ...

