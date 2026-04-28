from typing import Optional

from sona_ai.pipelines import SpeechPipeline


class TranscriptionService:
    def __init__(self, pipeline: SpeechPipeline):
        self.pipeline = pipeline

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ):
        return self.pipeline.transcribe(
            audio_path,
            language=language,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )

    def close(self):
        self.pipeline.cleanup_models()
