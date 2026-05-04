import re
from typing import Any

from sona_ai.core import sanitize_for_json, setup_logging
from sona_ai.diarization.schemas import DiarizationResult
from sona_ai.transcription.schemas import TranscriptSegment, TranscriptionResult, WordSegment


logger = setup_logging()


class WhisperXSpeakerAssigner:
    def assign(
        self,
        transcription: TranscriptionResult,
        diarization: DiarizationResult,
    ) -> list[dict[str, Any]]:
        if self._has_word_timestamps(transcription):
            logger.info("Assigning speakers by word/diarization overlap...")
            return self.assign_by_word_overlap(transcription, diarization)

        logger.info("Assigning speakers by segment/diarization overlap...")
        return self.assign_by_overlap(transcription, diarization)

    def assign_by_word_overlap(
        self,
        transcription: TranscriptionResult,
        diarization: DiarizationResult,
    ) -> list[dict[str, Any]]:
        assigned_segments = []
        previous_speaker = "Unknown"

        for segment in transcription.segments:
            segment_speaker = (
                self._speaker_for_interval(segment.start, segment.end, diarization)
                or previous_speaker
            )
            segment_groups = self._word_groups_for_segment(
                segment,
                diarization,
                segment_speaker,
            )

            if not segment_groups:
                segment_dict = segment.to_dict()
                segment_dict["speaker"] = segment_speaker
                assigned_segments.append(segment_dict)
                previous_speaker = segment_speaker
                continue

            assigned_segments.extend(segment_groups)
            previous_speaker = segment_groups[-1]["speaker"]

        return sanitize_for_json(assigned_segments)

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

    def _word_groups_for_segment(
        self,
        segment: TranscriptSegment,
        diarization: DiarizationResult,
        default_speaker: str,
    ) -> list[dict[str, Any]]:
        groups: list[dict[str, Any]] = []
        current_group: dict[str, Any] | None = None

        for word in segment.words:
            if not self._has_timing(word):
                if current_group is not None:
                    self._append_word(current_group, word, current_group["speaker"])
                continue

            speaker = (
                self._speaker_for_word(word, diarization)
                or default_speaker
            )

            if current_group is None or current_group["speaker"] != speaker:
                current_group = {
                    "speaker": speaker,
                    "start": word.start,
                    "end": word.end,
                    "words": [],
                }
                groups.append(current_group)
            else:
                current_group["end"] = word.end

            self._append_word(current_group, word, speaker)

        return [self._finalize_word_group(group) for group in groups]

    def _speaker_for_word(
        self,
        word: WordSegment,
        diarization: DiarizationResult,
    ) -> str | None:
        midpoint = (word.start + word.end) / 2
        containing_turns = [
            turn
            for turn in diarization.turns
            if turn.start <= midpoint <= turn.end
        ]
        if containing_turns:
            turn = min(containing_turns, key=lambda item: item.end - item.start)
            return turn.speaker

        return self._speaker_for_interval(word.start, word.end, diarization)

    def _append_word(
        self,
        group: dict[str, Any],
        word: WordSegment,
        speaker: str,
    ) -> None:
        word_dict = word.to_dict()
        word_dict["speaker"] = speaker
        group["words"].append(word_dict)

    def _finalize_word_group(self, group: dict[str, Any]) -> dict[str, Any]:
        return {
            "speaker": group["speaker"],
            "text": self._join_word_text(group["words"]),
            "start": group["start"],
            "end": group["end"],
            "words": group["words"],
        }

    def _join_word_text(self, words: list[dict[str, Any]]) -> str:
        tokens = [str(word.get("word", "")) for word in words if word.get("word")]
        if not tokens:
            return ""

        if any(token[:1].isspace() for token in tokens):
            text = "".join(tokens)
        else:
            text = " ".join(tokens)

        return re.sub(r"\s+([,.;:!?])", r"\1", text).strip()

    def _has_word_timestamps(self, transcription: TranscriptionResult) -> bool:
        return any(
            self._has_timing(word)
            for segment in transcription.segments
            for word in segment.words
        )

    def _has_timing(self, word: WordSegment) -> bool:
        return (
            word.start is not None
            and word.end is not None
            and word.end > word.start
        )
