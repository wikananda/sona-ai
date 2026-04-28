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

    def to_segment_dicts(self) -> list[dict[str, Any]]:
        return [segment.to_dict() for segment in self.segments]

