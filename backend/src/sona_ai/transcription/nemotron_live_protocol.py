import math
from copy import deepcopy
from typing import Any, Optional

from sona_ai.transcription.whisper_live_protocol import PROTOCOL_VERSION


class NemotronLiveProtocolError(ValueError):
    """Raised when a NeMo-Speech.cpp realtime event cannot be normalized."""


class NemotronLiveTranscriptAccumulator:
    """Translate append-only Nemotron deltas into Sona's revision protocol."""

    def __init__(self, session_id: str, language: Optional[str] = None):
        self.session_id = session_id
        self.language = language if language not in {None, "auto"} else None
        self.revision = 0
        self.audio_end = 0.0
        self._committed: list[dict[str, Any]] = []
        self._completed_items: set[str] = set()
        self._partial = ""
        self._partial_item_id: Optional[str] = None
        self._provisional: Optional[dict[str, Any]] = None

    @property
    def committed(self) -> list[dict[str, Any]]:
        return deepcopy(self._committed)

    def set_audio_end(self, value: float) -> None:
        self.audio_end = max(self.audio_end, _finite_time(value, self.audio_end))

    def apply_delta(self, message: Any) -> Optional[dict[str, Any]]:
        if not isinstance(message, dict):
            raise NemotronLiveProtocolError("Nemotron delta must be an object")
        delta = message.get("delta")
        if not isinstance(delta, str):
            raise NemotronLiveProtocolError("Nemotron delta text must be a string")
        if not delta:
            return None

        item_id = _item_id(message)
        if (
            self._partial_item_id is not None
            and item_id is not None
            and item_id != self._partial_item_id
            and self._partial
        ):
            raise NemotronLiveProtocolError("Nemotron changed items before completing the transcript")
        self._partial_item_id = item_id or self._partial_item_id
        self._partial += delta
        if len(self._partial) > 1_000_000:
            raise NemotronLiveProtocolError("Nemotron transcript exceeded the session limit")

        start = self._committed_end()
        self._provisional = {
            "text": self._partial.strip(),
            "start": start,
            "end": max(start, self.audio_end),
        }
        if not self._provisional["text"]:
            self._provisional = None
            return None

        self.revision += 1
        return self._event("transcript", [])

    def apply_completed(self, message: Any) -> Optional[dict[str, Any]]:
        if not isinstance(message, dict):
            raise NemotronLiveProtocolError("Nemotron completion must be an object")

        item_id = _item_id(message)
        if item_id is not None and item_id in self._completed_items:
            return None
        text = str(message.get("transcript") or message.get("text") or self._partial).strip()
        self._set_language_from_message(message)
        words = _normalize_words(message, committed_end=self._committed_end())

        committed: list[dict[str, Any]] = []
        if text:
            start = words[0]["start"] if words else self._committed_end()
            end = words[-1]["end"] if words else max(start, self.audio_end)
            segment: dict[str, Any] = {
                "text": text,
                "start": start,
                "end": max(start, end),
            }
            if words:
                segment["words"] = words
                speakers = {word.get("speaker") for word in words if word.get("speaker")}
                if len(speakers) == 1:
                    segment["speaker"] = speakers.pop()
            self._committed.append(segment)
            committed.append(deepcopy(segment))

        if item_id is not None:
            self._completed_items.add(item_id)
        self._partial = ""
        self._partial_item_id = None
        self._provisional = None
        if not committed and not self.language:
            return None

        self.revision += 1
        return self._event("transcript", committed)

    def finalize(self) -> dict[str, Any]:
        promoted: list[dict[str, Any]] = []
        if self._provisional is not None:
            provisional = deepcopy(self._provisional)
            self._committed.append(provisional)
            promoted.append(deepcopy(provisional))
            self._provisional = None
            self._partial = ""
            self._partial_item_id = None

        self.revision += 1
        event = self._event("final", promoted)
        event["segments"] = deepcopy(self._committed)
        return event

    def _set_language_from_message(self, message: dict[str, Any]) -> None:
        language = message.get("language")
        if not language:
            languages = message.get("languages")
            if isinstance(languages, list) and languages and isinstance(languages[0], dict):
                language = languages[0].get("code")
        normalized = str(language or "").strip()
        if normalized:
            self.language = normalized

    def _committed_end(self) -> float:
        return self._committed[-1]["end"] if self._committed else 0.0

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


def _normalize_words(
    message: dict[str, Any],
    *,
    committed_end: float,
) -> list[dict[str, Any]]:
    raw_words = message.get("words")
    if not isinstance(raw_words, list):
        words_info = message.get("words_info")
        raw_words = words_info.get("words", []) if isinstance(words_info, dict) else []

    parsed: list[dict[str, Any]] = []
    for raw in raw_words:
        if not isinstance(raw, dict):
            continue
        text = str(raw.get("word") or raw.get("text") or "").strip()
        start = _finite_number(raw.get("start", raw.get("start_time")))
        end = _finite_number(raw.get("end", raw.get("end_time")))
        if not text or start is None or end is None:
            continue
        start = max(0.0, start)
        end = max(start, end)
        word: dict[str, Any] = {"word": text, "start": start, "end": end}
        score = _finite_number(raw.get("confidence", raw.get("score")))
        if score is not None:
            word["score"] = min(max(score, 0.0), 1.0)
        speaker = raw.get("speaker", raw.get("speaker_tag"))
        if speaker is not None and str(speaker).strip():
            word["speaker"] = str(speaker)
        parsed.append(word)

    parsed.sort(key=lambda word: (word["start"], word["end"]))
    # Endpointed items may report timings relative to the item. Shift those
    # after already committed speech while leaving session-relative times alone.
    if parsed and parsed[0]["start"] < committed_end - 0.05:
        for word in parsed:
            word["start"] += committed_end
            word["end"] += committed_end
    return parsed


def _item_id(message: dict[str, Any]) -> Optional[str]:
    value = message.get("item_id")
    return str(value) if value is not None else None


def _finite_time(value: Any, default: float) -> float:
    number = _finite_number(value)
    return max(0.0, number if number is not None else default)


def _finite_number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
