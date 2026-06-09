import logging
import os
import warnings
from typing import Optional
from collections.abc import Callable

from sona_ai.alignment.base import Aligner
from sona_ai.core import PROJECT_ROOT, setup_logging, setup_model_cache_environment, write_json
from sona_ai.diarization.base import Diarizer
from sona_ai.pipelines.speaker_assignment import SpeakerAssigner
from sona_ai.transcription.base import Transcriber
from sona_ai.transcription.schemas import TranscriptionResult


logger = setup_logging()
output_dir = PROJECT_ROOT / "outputs" / "transcription"
ProgressCallback = Callable[[str, bool], None]


class SpeechPipeline:
    def __init__(
        self,
        transcriber: Transcriber,
        aligner: Optional[Aligner] = None,
        diarizer: Optional[Diarizer] = None,
        speaker_assigner: Optional[SpeakerAssigner] = None,
        write_outputs: bool = True,
    ):
        self.transcriber = transcriber
        self.aligner = aligner
        self.diarizer = diarizer
        self.speaker_assigner = speaker_assigner or SpeakerAssigner()
        self.write_outputs = write_outputs

    def load_models(self):
        self.transcriber.load_models()
        if self.aligner is not None:
            self.aligner.load_models()
        if self.diarizer is not None:
            self.diarizer.load_models()

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ):
        logger.info("Speech pipeline stage started: transcription")
        self._emit_progress(progress_callback, "transcribing", finished=False)
        transcription = self.transcriber.transcribe(audio_path, language=language)
        logger.info(
            "Speech pipeline stage finished: transcription (%d segments, %d timed segments, %d timed words)",
            len(transcription.segments),
            self._timed_segment_count(transcription),
            self._timed_word_count(transcription),
        )
        self._emit_progress(progress_callback, "transcribing", finished=True)
        if self.aligner is not None:
            logger.info("Speech pipeline stage started: alignment")
            self._emit_progress(progress_callback, "aligning", finished=False)
            transcription = self.aligner.align(transcription, audio_path)
            logger.info(
                "Speech pipeline stage finished: alignment (%d segments, %d timed segments, %d timed words)",
                len(transcription.segments),
                self._timed_segment_count(transcription),
                self._timed_word_count(transcription),
            )
            self._emit_progress(progress_callback, "aligning", finished=True)

        if self.diarizer is None:
            segments = transcription.to_segment_dicts()
            conversations = self._build_conversations(segments)
            result = {
                "transcript": conversations,
                "diarize_result": [],
                "result_raw": segments,
            }
            self._write_result(result)
            return result

        return self.extract_speakers(
            audio_path,
            transcription,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            progress_callback=progress_callback,
        )

    def extract_speakers(
        self,
        audio_path: str,
        transcription: TranscriptionResult,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ):
        if self.diarizer is None:
            raise ValueError("Speaker extraction is not available because diarization is disabled")
        if not self._has_usable_timestamps(transcription):
            raise ValueError(
                "Cannot assign speakers because transcript has no usable timestamps. "
                "Re-transcribe with alignment enabled."
            )

        logger.info("Speech pipeline stage started: diarization")
        self._emit_progress(progress_callback, "diarizing", finished=False)
        diarization = self.diarizer.diarize(
            audio_path,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
        logger.info(
            "Speech pipeline stage finished: diarization (%d turns, %d speakers, min_speakers=%s max_speakers=%s)",
            len(diarization.turns),
            len({turn.speaker for turn in diarization.turns}),
            min_speakers,
            max_speakers,
        )
        self._emit_progress(progress_callback, "diarizing", finished=True)
        logger.info("Speech pipeline stage started: speaker assignment")
        self._emit_progress(progress_callback, "assigning_speakers", finished=False)
        segments = self.speaker_assigner.assign(transcription, diarization)
        speakers = sorted(self._real_speakers(segments))
        logger.info(
            "Final transcript has %d speakers across %d segments: %s",
            len(speakers),
            len(segments),
            speakers,
        )
        if not speakers:
            raise ValueError(
                "Speaker extraction did not assign any speakers. The transcript may "
                "not have enough timing detail. Re-transcribe with alignment enabled."
            )
        logger.info("Speech pipeline stage finished: speaker assignment")
        self._emit_progress(progress_callback, "assigning_speakers", finished=True)
        conversations = self._build_conversations(segments)

        result = {
            "transcript": conversations,
            "diarize_result": diarization.to_dict(),
            "result_raw": segments,
        }
        self._write_result(result)
        return result

    def _build_conversations(self, result_segments):
        conversations = []

        for segment in result_segments:
            conversation = {
                "text": segment["text"],
                "start": segment["start"],
                "end": segment["end"],
            }
            if segment.get("speaker"):
                conversation["speaker"] = segment["speaker"]
            conversations.append(conversation)

        return conversations

    def _write_result(self, result):
        if not self.write_outputs:
            return

        write_json(output_dir / "conversations.json", result["transcript"])
        write_json(output_dir / "result_raw.json", result["result_raw"])

    def _emit_progress(
        self,
        progress_callback: Optional[ProgressCallback],
        stage: str,
        finished: bool,
    ) -> None:
        if progress_callback is None:
            return
        progress_callback(stage, finished)

    def _has_usable_timestamps(self, transcription: TranscriptionResult) -> bool:
        return (
            self._timed_word_count(transcription) > 0
            or self._timed_segment_count(transcription) > 0
        )

    def _timed_segment_count(self, transcription: TranscriptionResult) -> int:
        return sum(
            1
            for segment in transcription.segments
            if segment.end > segment.start
        )

    def _timed_word_count(self, transcription: TranscriptionResult) -> int:
        return sum(
            1
            for segment in transcription.segments
            for word in segment.words
            if (
                word.start is not None
                and word.end is not None
                and word.end > word.start
            )
        )

    def _real_speakers(self, segments: list[dict]) -> set[str]:
        return {
            str(segment.get("speaker")).strip()
            for segment in segments
            if isinstance(segment, dict)
            and segment.get("speaker")
            and str(segment.get("speaker")).strip().lower() != "unknown"
        }

    def cleanup_models(self):
        self.transcriber.cleanup_models()
        if self.aligner is not None:
            self.aligner.cleanup_models()
        if self.diarizer is not None:
            self.diarizer.cleanup_models()

    @staticmethod
    def setup_environment(config: dict = None, quiet=False):
        if quiet:
            warnings.filterwarnings("ignore")
            logging.getLogger("faster_whisper").setLevel(logging.ERROR)

        os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

        setup_model_cache_environment(config)

    def close(self):
        self.cleanup_models()
