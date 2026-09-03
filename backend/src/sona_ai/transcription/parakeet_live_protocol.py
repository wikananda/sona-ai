import math
import re
from copy import deepcopy
from typing import Any, Optional

from sona_ai.transcription.whisper_live_protocol import PROTOCOL_VERSION


_TOKEN_NORMALIZER = re.compile(r"[^\w]+", re.UNICODE)
_TIMESTAMP_JITTER_SECONDS = 0.35


class ParakeetLiveProtocolError(ValueError):
    """Raised when a buffered Parakeet hypothesis cannot be normalized safely."""


class ParakeetLiveTranscriptAccumulator:
    """Convert overlapping Parakeet snapshots into stable transcript deltas.

    Parakeet TDT is decoded over a rolling context window. Words at or before
    ``stable_cutoff`` become immutable while the newer right-context tail stays
    provisional and may be replaced by the next snapshot.
    """

    def __init__(self, session_id: str, language: Optional[str] = None):
        self.session_id = session_id
        self.language = language
        self.revision = 0
        self._committed: list[dict[str, Any]] = []
        self._committed_words: list[dict[str, Any]] = []
        self._commit_horizon = 0.0
        self._provisional: Optional[dict[str, Any]] = None
        self._pending_stable_words: list[dict[str, Any]] = []

    @property
    def committed(self) -> list[dict[str, Any]]:
        return deepcopy(self._committed)

    @property
    def commit_horizon(self) -> float:
        return self._commit_horizon

    def apply_snapshot(
        self,
        raw_segments: Any,
        *,
        window_start: float,
        stable_cutoff: float,
        audio_end: float,
        final: bool = False,
    ) -> Optional[dict[str, Any]]:
        if not isinstance(raw_segments, list):
            raise ParakeetLiveProtocolError("Parakeet segments must be a list")
        if window_start < 0 or stable_cutoff < 0 or audio_end < 0:
            raise ParakeetLiveProtocolError("Parakeet snapshot times must be non-negative")
        if stable_cutoff > audio_end:
            raise ParakeetLiveProtocolError("Parakeet stable cutoff exceeds captured audio")

        words = _snapshot_words(
            raw_segments,
            window_start=window_start,
            audio_end=audio_end,
        )
        candidates = [
            word
            for word in words
            if word["end"] > self._commit_horizon - _TIMESTAMP_JITTER_SECONDS
        ]
        candidates = _drop_committed_overlap(self._committed_words, candidates)
        candidates = _deduplicate_adjacent(candidates)
        stable_words = [
            word for word in candidates if word["end"] <= stable_cutoff + 0.02
        ]

        # A full-context model can revise even an apparently old token as more
        # speech arrives. Commit only the stable prefix that agreed while stable
        # in the preceding snapshot. The final decode is authoritative.
        confirmed_count = (
            len(stable_words)
            if final
            else _matching_prefix_length(self._pending_stable_words, stable_words)
        )
        confirmed_words = stable_words[:confirmed_count]
        pending_stable_words = stable_words[confirmed_count:]
        newly_committed: list[dict[str, Any]] = []
        if confirmed_words:
            committed_segment = _segment_from_words(confirmed_words)
            self._committed.append(committed_segment)
            self._committed_words.extend(deepcopy(confirmed_words))
            newly_committed.append(deepcopy(committed_segment))

        # Do not age out a stable token until it has had another snapshot in
        # which to prove itself. With no pending token, advancing through
        # silence prevents an old rolling-window word from reappearing.
        safe_horizon = (
            min(word["start"] for word in pending_stable_words)
            if pending_stable_words
            else stable_cutoff
        )
        self._commit_horizon = max(
            self._commit_horizon,
            min(stable_cutoff, safe_horizon),
        )
        self._pending_stable_words = deepcopy(pending_stable_words)

        provisional_words = candidates[confirmed_count:]
        next_provisional = (
            _segment_from_words(provisional_words)
            if provisional_words
            else None
        )

        provisional_changed = next_provisional != self._provisional
        self._provisional = next_provisional
        if not newly_committed and not provisional_changed:
            return None

        self.revision += 1
        return self._event("transcript", newly_committed)

    def finalize(self) -> dict[str, Any]:
        promoted: list[dict[str, Any]] = []
        if self._provisional is not None:
            provisional = self._provisional
            self._committed.append(provisional)
            self._committed_words.extend(deepcopy(provisional.get("words", [])))
            promoted.append(deepcopy(provisional))
            self._provisional = None

        self.revision += 1
        event = self._event("final", promoted)
        event["segments"] = deepcopy(self._committed)
        return event

    def _event(
        self,
        event_type: str,
        committed: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "type": event_type,
            "version": PROTOCOL_VERSION,
            "session_id": self.session_id,
            "revision": self.revision,
            "committed": committed,
            "provisional": deepcopy(self._provisional),
            "language": self.language,
        }


def _snapshot_words(
    raw_segments: list[Any],
    *,
    window_start: float,
    audio_end: float,
) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    for raw_segment in raw_segments:
        if not isinstance(raw_segment, dict):
            continue
        for raw_word in raw_segment.get("words") or []:
            word = _normalize_word(
                raw_word,
                window_start=window_start,
                audio_end=audio_end,
            )
            if word is not None:
                words.append(word)
    words.sort(key=lambda item: (item["start"], item["end"]))
    return words


def _normalize_word(
    raw: Any,
    *,
    window_start: float,
    audio_end: float,
) -> Optional[dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    text = str(raw.get("word") or "").strip()
    start = _finite_number(raw.get("start"))
    end = _finite_number(raw.get("end"))
    if not text or start is None or end is None:
        return None

    start = min(max(window_start + start, window_start), audio_end)
    end = min(max(window_start + end, start), audio_end)
    if end <= start:
        return None

    word: dict[str, Any] = {
        "word": text,
        "start": start,
        "end": end,
    }
    score = _finite_number(raw.get("score", raw.get("confidence")))
    if score is not None:
        word["score"] = min(max(score, 0.0), 1.0)
    speaker = raw.get("speaker")
    if speaker is not None and str(speaker).strip():
        word["speaker"] = str(speaker)
    return word


def _drop_committed_overlap(
    committed: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not committed or not candidates:
        return candidates
    max_overlap = min(8, len(committed), len(candidates))
    for overlap in range(max_overlap, 0, -1):
        committed_tail = [
            _normalized_token(word["word"])
            for word in committed[-overlap:]
        ]
        candidate_head = [
            _normalized_token(word["word"])
            for word in candidates[:overlap]
        ]
        if committed_tail == candidate_head:
            return candidates[overlap:]
    return candidates


def _deduplicate_adjacent(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduplicated: list[dict[str, Any]] = []
    for word in words:
        if deduplicated:
            previous = deduplicated[-1]
            same_token = _normalized_token(previous["word"]) == _normalized_token(word["word"])
            same_time = abs(previous["start"] - word["start"]) <= 0.05
            if same_token and same_time:
                continue
        deduplicated.append(word)
    return deduplicated


def _matching_prefix_length(
    previous: list[dict[str, Any]],
    current: list[dict[str, Any]],
) -> int:
    matched = 0
    for previous_word, current_word in zip(previous, current):
        same_token = (
            _normalized_token(previous_word["word"])
            == _normalized_token(current_word["word"])
        )
        same_start = (
            abs(previous_word["start"] - current_word["start"])
            <= _TIMESTAMP_JITTER_SECONDS
        )
        same_end = (
            abs(previous_word["end"] - current_word["end"])
            <= _TIMESTAMP_JITTER_SECONDS
        )
        if not (same_token and same_start and same_end):
            break
        matched += 1
    return matched


def _segment_from_words(words: list[dict[str, Any]]) -> dict[str, Any]:
    segment: dict[str, Any] = {
        "text": " ".join(word["word"] for word in words),
        "start": words[0]["start"],
        "end": words[-1]["end"],
        "words": deepcopy(words),
    }
    speakers = {word.get("speaker") for word in words if word.get("speaker")}
    if len(speakers) == 1:
        segment["speaker"] = speakers.pop()
    return segment


def _normalized_token(value: str) -> str:
    return _TOKEN_NORMALIZER.sub("", value.casefold())


def _finite_number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
