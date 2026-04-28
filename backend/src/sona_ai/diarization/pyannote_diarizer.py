import gc
import os
from typing import Optional

import torch
import whisperx
from dotenv import load_dotenv
from whisperx.diarize import DiarizationPipeline

from sona_ai.core import PROJECT_ROOT, Timer, setup_logging
from sona_ai.diarization.schemas import DiarizationResult, SpeakerTurn


load_dotenv(PROJECT_ROOT / ".env")
logger = setup_logging()


class PyannoteDiarizer:
    def __init__(self, config: dict):
        self.config = config
        self.model = None

    def load_models(self):
        logger.info("Loading pyannote diarization model...")
        if os.getenv("HF_TOKEN") is None:
            raise EnvironmentError("Hugging Face token is not set")

        self.model = DiarizationPipeline(
            use_auth_token=os.getenv("HF_TOKEN"),
            device=self.config["model"]["device"],
        )

    def diarize(
        self,
        audio_path: str,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> DiarizationResult:
        if self.model is None:
            raise ReferenceError("Diarization model is not initialized.")

        min_s, max_s = self._resolve_speaker_bounds(min_speakers, max_speakers)
        audio = whisperx.load_audio(audio_path)

        logger.info("Running diarization...")
        with Timer("Diarization"):
            diarization = self.model(
                audio,
                min_speakers=min_s,
                max_speakers=max_s,
            )

        return DiarizationResult(
            turns=self._build_turns(diarization),
            raw=diarization,
        )

    def _resolve_speaker_bounds(
        self,
        min_speakers: Optional[int],
        max_speakers: Optional[int],
    ) -> tuple[int, int]:
        min_s = min_speakers if min_speakers is not None else self.config["input"]["min_speakers"]
        max_s = max_speakers if max_speakers is not None else self.config["input"]["max_speakers"]

        if min_s > max_s:
            raise ValueError("min_speakers must be less than or equal to max_speakers")
        if min_s < 1:
            raise ValueError("min_speakers must be greater than or equal to 1")
        if max_s < 1:
            raise ValueError("max_speakers must be greater than or equal to 1")

        return min_s, max_s

    def _build_turns(self, diarization) -> list[SpeakerTurn]:
        turns = []
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            turns.append(
                SpeakerTurn(
                    start=segment.start,
                    end=segment.end,
                    speaker=str(speaker),
                )
            )
        return turns

    def cleanup_models(self):
        if self.model is not None:
            del self.model
            self.model = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

