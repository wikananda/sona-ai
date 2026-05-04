from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class WordSegment:
    word: str
    start: Optional[float] = None
    end: Optional[float] = None
    score: Optional[float] = None
    speaker: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WordSegment":
        return cls(
            word=data.get("word", ""),
            start=data.get("start"),
            end=data.get("end"),
            score=data.get("score"),
            speaker=data.get("speaker"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "word": self.word,
            "start": self.start,
            "end": self.end,
            "score": self.score,
            "speaker": self.speaker,
        }


@dataclass(frozen=True)
class TranscriptSegment:
    text: str
    start: float
    end: float
    speaker: Optional[str] = None
    words: list[WordSegment] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TranscriptSegment":
        words = [WordSegment.from_dict(word) for word in data.get("words", [])]
        return cls(
            text=data.get("text", ""),
            start=data.get("start", 0.0),
            end=data.get("end", 0.0),
            speaker=data.get("speaker"),
            words=words,
        )

    def to_dict(self) -> dict[str, Any]:
        data = {
            "text": self.text,
            "start": self.start,
            "end": self.end,
        }
        if self.speaker is not None:
            data["speaker"] = self.speaker
        if self.words:
            data["words"] = [word.to_dict() for word in self.words]
        return data


@dataclass
class TranscriptionResult:
    segments: list[TranscriptSegment]
    language: Optional[str] = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_whisperx_result(cls, result: dict[str, Any]) -> "TranscriptionResult":
        return cls(
            segments=[
                TranscriptSegment.from_dict(segment)
                for segment in result.get("segments", [])
            ],
            language=result.get("language"),
            raw=result,
        )

    @classmethod
    def from_parakeet_hypothesis(
        cls,
        hypothesis: Any,
        language: Optional[str] = None,
    ) -> "TranscriptionResult":
        text = getattr(hypothesis, "text", None) or str(hypothesis)
        timestamps = getattr(hypothesis, "timestamp", None) or {}
        word_timestamps = timestamps.get("word") or []
        segment_timestamps = timestamps.get("segment") or []

        words = [_word_from_stamp(stamp) for stamp in word_timestamps]
        segments = _segments_from_stamps(text, segment_timestamps, words)
        raw = {
            "text": text,
            "segments": [segment.to_dict() for segment in segments],
            "timestamp": timestamps,
        }

        return cls(
            segments=segments,
            language=language,
            raw=raw,
        )

    def to_segment_dicts(self) -> list[dict[str, Any]]:
        return [segment.to_dict() for segment in self.segments]


def _word_from_stamp(stamp: dict[str, Any]) -> WordSegment:
    return WordSegment(
        word=str(_stamp_text(stamp, ["word", "text", "token", "char"])),
        start=_stamp_time(stamp, ["start", "start_offset"]),
        end=_stamp_time(stamp, ["end", "end_offset"]),
    )


def _segments_from_stamps(
    text: str,
    segment_timestamps: list[dict[str, Any]],
    words: list[WordSegment],
) -> list[TranscriptSegment]:
    if segment_timestamps:
        return [
            _segment_from_stamp(stamp, words)
            for stamp in segment_timestamps
        ]

    start = words[0].start if words and words[0].start is not None else 0.0
    end = words[-1].end if words and words[-1].end is not None else start

    return [
        TranscriptSegment(
            text=text,
            start=start,
            end=end,
            words=words,
        )
    ]


def _segment_from_stamp(
    stamp: dict[str, Any],
    words: list[WordSegment],
) -> TranscriptSegment:
    text = str(_stamp_text(stamp, ["segment", "text", "word"]))
    start = _stamp_time(stamp, ["start", "start_offset"]) or 0.0
    end = _stamp_time(stamp, ["end", "end_offset"]) or start
    segment_words = [
        word
        for word in words
        if _word_overlaps_segment(word, start, end)
    ]

    return TranscriptSegment(
        text=text,
        start=start,
        end=end,
        words=segment_words,
    )


def _stamp_text(stamp: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        if key in stamp and stamp[key] is not None:
            return stamp[key]
    return ""


def _stamp_time(stamp: dict[str, Any], keys: list[str]) -> Optional[float]:
    for key in keys:
        if key in stamp and stamp[key] is not None:
            return float(stamp[key])
    return None


def _word_overlaps_segment(
    word: WordSegment,
    segment_start: float,
    segment_end: float,
) -> bool:
    if word.start is None or word.end is None:
        return False
    return max(0.0, min(segment_end, word.end) - max(segment_start, word.start)) > 0
