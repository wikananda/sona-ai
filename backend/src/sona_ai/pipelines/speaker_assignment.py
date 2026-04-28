from typing import Any

from sona_ai.core import Timer, sanitize_for_json, setup_logging
from sona_ai.diarization.schemas import DiarizationResult
from sona_ai.transcription.schemas import TranscriptionResult


logger = setup_logging()


class WhisperXSpeakerAssigner:
    def assign(
        self,
        transcription: TranscriptionResult,
        diarization: DiarizationResult,
    ) -> list[dict[str, Any]]:
        if diarization.raw is not None and "segments" in transcription.raw:
            try:
                import whisperx

                logger.info("Assigning speakers with WhisperX...")
                with Timer("Assign diarization to transcription"):
                    assigned = whisperx.assign_word_speakers(
                        diarization.raw,
                        transcription.raw,
                    )
                return sanitize_for_json(assigned["segments"])
            except Exception as exc:
                logger.warning("WhisperX speaker assignment failed: %s", exc)

        return self.assign_by_overlap(transcription, diarization)

    def assign_by_overlap(
        self,
        transcription: TranscriptionResult,
        diarization: DiarizationResult,
    ) -> list[dict[str, Any]]:
        assigned_segments = []
        for segment in transcription.segments:
            speaker = self._speaker_for_interval(segment.start, segment.end, diarization)
            segment_dict = segment.to_dict()
            segment_dict["speaker"] = speaker or "Unknown"
            assigned_segments.append(segment_dict)

        return sanitize_for_json(assigned_segments)

    def _speaker_for_interval(
        self,
        start: float,
        end: float,
        diarization: DiarizationResult,
    ) -> str | None:
        scores: dict[str, float] = {}
        for turn in diarization.turns:
            overlap = max(0.0, min(end, turn.end) - max(start, turn.start))
            if overlap > 0:
                scores[turn.speaker] = scores.get(turn.speaker, 0.0) + overlap

        if not scores:
            return None

        return max(scores, key=scores.get)
