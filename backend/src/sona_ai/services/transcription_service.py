from typing import Optional

from sona_ai.transcription import WhisperXEngine


class TranscriptionService:
    def __init__(self, engine: WhisperXEngine):
        self.engine = engine

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ):
        return self.engine.transcribe(
            audio_path,
            language=language,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )

    def close(self):
        self.engine.cleanup_models()

