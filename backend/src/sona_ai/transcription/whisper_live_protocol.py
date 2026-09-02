import math
from copy import deepcopy
from typing import Any, Optional


PROTOCOL_VERSION = 1


class WhisperLiveProtocolError(ValueError):
    """Raised when WhisperLive sends a response that cannot be normalized safely."""


class WhisperLiveTranscriptAccumulator:
    """Turn WhisperLive's rolling snapshots into stable transcript deltas."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.language: Optional[str] = None
        self.revision = 0
        self._committed: list[dict[str, Any]] = []
        self._committed_keys: set[tuple[float, float, str]] = set()
        self._provisional: Optional[dict[str, Any]] = None

    @property
    def committed(self) -> list[dict[str, Any]]:
        return deepcopy(self._committed)

    @property
    def provisional(self) -> Optional[dict[str, Any]]:
        return deepcopy(self._provisional)

    def set_language(self, language: Any) -> Optional[dict[str, Any]]:
        normalized = str(language or "").strip() or None
        if normalized == self.language:
            return None
        self.language = normalized
        self.revision += 1
        return self._event("transcript", committed=[])

    def apply_segments(self, raw_segments: Any) -> Optional[dict[str, Any]]:
        if not isinstance(raw_segments, list):
            raise WhisperLiveProtocolError("WhisperLive segments must be a list")

        normalized = [
            segment
            for raw in raw_segments
            if (segment := _normalize_segment(raw)) is not None
        ]
        newly_committed = []
        next_provisional = None

        for segment in normalized:
            if segment.pop("completed", False):
                key = _segment_key(segment)
                if key in self._committed_keys:
                    continue
                if self._committed and segment["start"] < self._committed[-1]["end"] - 0.05:
                    continue
                self._committed_keys.add(key)
                self._committed.append(segment)
                newly_committed.append(deepcopy(segment))
            else:
                next_provisional = segment

        if next_provisional is not None and self._committed:
            if next_provisional["end"] <= self._committed[-1]["end"]:
                next_provisional = None
            else:
                next_provisional["start"] = max(
                    next_provisional["start"],
                    self._committed[-1]["end"],
                )
                next_provisional["end"] = max(
                    next_provisional["end"],
                    next_provisional["start"],
                )

        provisional_changed = next_provisional != self._provisional
        self._provisional = next_provisional
        if not newly_committed and not provisional_changed:
            return None

        self.revision += 1
        return self._event("transcript", committed=newly_committed)

    def finalize(self) -> dict[str, Any]:
        promoted = []
        if self._provisional is not None:
            provisional = self._provisional
            key = _segment_key(provisional)
            if key not in self._committed_keys:
                self._committed_keys.add(key)
                self._committed.append(provisional)
                promoted.append(deepcopy(provisional))
            self._provisional = None

        self.revision += 1
        event = self._event("final", committed=promoted)
        event["segments"] = deepcopy(self._committed)
        return event

    def _event(self, event_type: str, committed: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "type": event_type,
            "version": PROTOCOL_VERSION,
            "session_id": self.session_id,
            "revision": self.revision,
            "committed": committed,
            "provisional": deepcopy(self._provisional),
            "language": self.language,
        }


def _normalize_segment(raw: Any) -> Optional[dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    text = str(raw.get("text") or "").strip()
    if not text:
        return None

    start = _finite_time(raw.get("start"), default=0.0)
    end = max(_finite_time(raw.get("end"), default=start), start)
    words = []
    for raw_word in raw.get("words") or []:
        word = _normalize_word(raw_word, segment_start=start, segment_end=end)
        if word is not None:
            words.append(word)

    segment: dict[str, Any] = {
        "text": text,
        "start": start,
        "end": end,
        "completed": bool(raw.get("completed", False)),
    }
    if words:
        segment["words"] = words
    speaker = str(raw.get("speaker") or "").strip()
    if speaker:
        segment["speaker"] = speaker
    return segment


def _normalize_word(
    raw: Any,
    segment_start: float,
    segment_end: float,
) -> Optional[dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    text = str(raw.get("word") or "")
    if not text.strip():
        return None

    start = min(
        max(_finite_time(raw.get("start"), default=segment_start), segment_start),
        segment_end,
    )
    end = min(
        max(_finite_time(raw.get("end"), default=start), start),
        segment_end,
    )
    word: dict[str, Any] = {"word": text, "start": start, "end": end}
    probability = _finite_number(raw.get("probability"))
    if probability is not None:
        word["score"] = min(max(probability, 0.0), 1.0)
    return word


def _finite_time(value: Any, default: float) -> float:
    number = _finite_number(value)
    return max(number if number is not None else default, 0.0)


def _finite_number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _segment_key(segment: dict[str, Any]) -> tuple[float, float, str]:
    return (
        round(segment["start"], 3),
        round(segment["end"], 3),
        segment["text"],
    )
