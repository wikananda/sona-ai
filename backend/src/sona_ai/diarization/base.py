from typing import Optional, Protocol

from .schemas import DiarizationResult


class Diarizer(Protocol):
    def load_models(self) -> None:
        ...

    def diarize(
        self,
        audio_path: str,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> DiarizationResult:
        ...

    def cleanup_models(self) -> None:
        ...

