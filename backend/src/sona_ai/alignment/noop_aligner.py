from sona_ai.transcription.schemas import TranscriptionResult

class NoOpAligner:
    def __init__(self, config: dict | None = None):
        self.config = config or {}

    def load_models(self) -> None:
        return None

    def align(
        self,
        transcription: TranscriptionResult,
        audio_path: str,
    ) -> TranscriptionResult:
        return transcription

    def cleanup_models(self) -> None:
        return None

    