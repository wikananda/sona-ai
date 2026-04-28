from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class SpeakerTurn:
    start: float
    end: float
    speaker: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "speaker": self.speaker,
        }


@dataclass
class DiarizationResult:
    turns: list[SpeakerTurn]
    raw: Optional[Any] = None

    def to_dict(self) -> list[dict[str, Any]]:
        return [turn.to_dict() for turn in self.turns]

