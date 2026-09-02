from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .summarization_service import SummarizationService
    from .transcription_service import TranscriptionService

__all__ = ["SummarizationService", "TranscriptionService"]


def __getattr__(name: str):
    if name == "SummarizationService":
        from .summarization_service import SummarizationService

        return SummarizationService
    if name == "TranscriptionService":
        from .transcription_service import TranscriptionService

        return TranscriptionService
    raise AttributeError(name)
