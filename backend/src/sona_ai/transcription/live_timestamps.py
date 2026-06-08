import re
import wave
from pathlib import Path

from sona_ai.transcription.schemas import (
    TranscriptSegment,
    TranscriptionResult,
    WordSegment,
)


MAX_WORDS_PER_SEGMENT = 22


def segment_live_timestamps(
    transcription: TranscriptionResult,
    audio_path: str,
) -> TranscriptionResult:
    """Turn one large live ASR segment into readable timed transcript segments."""
    if _has_multiple_usable_segments(transcription.segments):
        return transcription

    duration = _audio_duration(audio_path)
    segments = []
    for segment in transcription.segments:
        segments.extend(_split_segment(segment, duration))

    if not segments:
        return transcription

    return TranscriptionResult(
        segments=segments,
        language=transcription.language,
        raw={
            **transcription.raw,
            "live_timestamping": {
                "mode": "rough_sentence_split",
                "source_segments": len(transcription.segments),
                "segments": len(segments),
            },
        },
    )


def _has_multiple_usable_segments(segments: list[TranscriptSegment]) -> bool:
    usable = [
        segment
        for segment in segments
        if segment.text.strip() and segment.end > segment.start
    ]
    return len(usable) > 1


def _split_segment(
    segment: TranscriptSegment,
    audio_duration: float,
) -> list[TranscriptSegment]:
    text = segment.text.strip()
    if not text:
        return []

    timed_words = [
        word
        for word in segment.words
        if word.word.strip()
        and word.start is not None
        and word.end is not None
        and word.end > word.start
    ]
    if timed_words:
        return _segments_from_timed_words(timed_words)

    start = segment.start if segment.end > segment.start else 0.0
    end = segment.end if segment.end > segment.start else audio_duration
    if end <= start:
        end = start + max(1.0, len(_words(text)) * 0.35)

    return _segments_from_text(text, start, end)


def _segments_from_timed_words(words: list[WordSegment]) -> list[TranscriptSegment]:
    grouped_words = _group_words(words)
    segments = []
    for group in grouped_words:
        start = float(group[0].start or 0.0)
        end = float(group[-1].end or start)
        segments.append(
            TranscriptSegment(
                text=_join_word_text([word.word for word in group]),
                start=start,
                end=max(end, start),
                words=group,
            )
        )
    return segments


def _segments_from_text(
    text: str,
    start: float,
    end: float,
) -> list[TranscriptSegment]:
    chunks = _text_chunks(text)
    if len(chunks) <= 1:
        return [TranscriptSegment(text=text, start=start, end=max(end, start))]

    total_words = sum(max(1, len(_words(chunk))) for chunk in chunks)
    cursor = start
    duration = max(0.0, end - start)
    segments = []
    for index, chunk in enumerate(chunks):
        word_count = max(1, len(_words(chunk)))
        if index == len(chunks) - 1:
            chunk_end = end
        else:
            chunk_end = cursor + duration * (word_count / total_words)
        segments.append(
            TranscriptSegment(
                text=chunk,
                start=cursor,
                end=max(chunk_end, cursor),
            )
        )
        cursor = chunk_end
    return segments


def _group_words(words: list[WordSegment]) -> list[list[WordSegment]]:
    groups = []
    current = []
    for word in words:
        current.append(word)
        word_text = word.word.strip()
        if _ends_sentence(word_text) or len(current) >= MAX_WORDS_PER_SEGMENT:
            groups.append(current)
            current = []

    if current:
        groups.append(current)

    return groups


def _text_chunks(text: str) -> list[str]:
    sentence_chunks = [
        chunk.strip()
        for chunk in re.findall(r"[^.!?]+(?:[.!?]+|$)", text)
        if chunk.strip()
    ]
    if len(sentence_chunks) > 1:
        return _limit_chunk_sizes(sentence_chunks)

    words = _words(text)
    if len(words) <= MAX_WORDS_PER_SEGMENT:
        return [text.strip()]

    return [
        _join_word_text(words[index:index + MAX_WORDS_PER_SEGMENT])
        for index in range(0, len(words), MAX_WORDS_PER_SEGMENT)
    ]


def _limit_chunk_sizes(chunks: list[str]) -> list[str]:
    limited = []
    for chunk in chunks:
        words = _words(chunk)
        if len(words) <= MAX_WORDS_PER_SEGMENT:
            limited.append(chunk)
            continue
        limited.extend(
            _join_word_text(words[index:index + MAX_WORDS_PER_SEGMENT])
            for index in range(0, len(words), MAX_WORDS_PER_SEGMENT)
        )
    return limited


def _words(text: str) -> list[str]:
    return re.findall(r"\S+", text.strip())


def _join_word_text(words: list[str]) -> str:
    text = " ".join(word.strip() for word in words if word.strip())
    return re.sub(r"\s+([,.;:!?])", r"\1", text).strip()


def _ends_sentence(word: str) -> bool:
    return bool(re.search(r'[.!?]["\')\]]*$', word))


def _audio_duration(audio_path: str) -> float:
    try:
        import soundfile as sf

        info = sf.info(audio_path)
        return float(info.duration or 0.0)
    except Exception:
        pass

    try:
        with wave.open(str(Path(audio_path)), "rb") as audio:
            rate = audio.getframerate()
            if rate > 0:
                return audio.getnframes() / float(rate)
    except Exception:
        return 0.0

    return 0.0
