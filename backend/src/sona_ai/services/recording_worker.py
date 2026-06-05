import json
import uuid

from sqlalchemy.orm import Session

from sona_ai.core import PROJECT_ROOT, sanitize_for_json, setup_logging
from sona_ai.db.engine import SessionLocal
from sona_ai.db.models import Recording, RecordingStatus, Transcript
from sona_ai.services.transcription_service import TranscriptionService
from sona_ai.transcription.schemas import TranscriptSegment, TranscriptionResult


logger = setup_logging()


def run_transcription(
    recording_id: str,
    transcription_service: TranscriptionService,
    extract_speakers: bool = True,
) -> None:
    logger.info("Recording worker started for recording_id=%s", recording_id)
    db = SessionLocal()
    try:
        recording = db.get(Recording, recording_id)
        if recording is None:
            logger.warning("Recording worker skipped missing recording_id=%s", recording_id)
            return

        _set_status(db, recording, RecordingStatus.PROCESSING)
        logger.info(
            "Recording %s marked processing: file=%s model=%s device=%s language=%s",
            recording.id,
            recording.stored_path,
            recording.model,
            recording.device,
            recording.language_hint,
        )

        profile = transcription_service.resolve_profile(
            model=recording.model,
            device=recording.device,
            extract_speakers=extract_speakers,
        )
        logger.info(
            "Resolved recording %s profile: transcription=%s alignment=%s diarization=%s",
            recording.id,
            profile.transcription_engine,
            profile.alignment_engine if profile.alignment_enabled else "disabled",
            profile.diarization_engine if profile.diarization_enabled else "disabled",
        )

        result = transcription_service.transcribe(
            str(PROJECT_ROOT / recording.stored_path),
            language=recording.language_hint,
            model=recording.model,
            device=recording.device,
            min_speakers=recording.min_speakers,
            max_speakers=recording.max_speakers,
            extract_speakers=extract_speakers,
        )

        transcript_segments = sanitize_for_json(
            result.get("result_raw") or result.get("transcript", [])
        )
        transcript = Transcript(
            id=str(uuid.uuid4()),
            recording_id=recording.id,
            segments_json=json.dumps(transcript_segments),
            language=recording.language_hint,
            transcription_engine=profile.transcription_engine,
            diarization_engine=(
                profile.diarization_engine if profile.diarization_enabled
                else None
            ),
            model_config_json=json.dumps(_transcript_metadata(profile, recording)),
        )

        if recording.transcript is not None:
            db.delete(recording.transcript)
            db.flush()

        db.add(transcript)
        recording.status = RecordingStatus.DONE
        recording.error = None
        db.commit()
        logger.info("Recording worker finished recording_id=%s", recording_id)
    except Exception as exc:
        logger.exception("Recording transcription failed: %s", exc)
        db.rollback()
        _mark_failed(db, recording_id, str(exc))
    finally:
        db.close()


def run_speaker_extraction(
    recording_id: str,
    transcription_service: TranscriptionService,
) -> None:
    logger.info("Speaker extraction worker started for recording_id=%s", recording_id)
    db = SessionLocal()
    try:
        recording = db.get(Recording, recording_id)
        if recording is None:
            logger.warning("Speaker extraction skipped missing recording_id=%s", recording_id)
            return
        if recording.transcript is None:
            raise ValueError("Transcript not found")

        _set_status(db, recording, RecordingStatus.PROCESSING)

        profile = transcription_service.resolve_profile(
            model=recording.model,
            device=recording.device,
            extract_speakers=True,
        )
        segments = json.loads(recording.transcript.segments_json)
        transcription = TranscriptionResult(
            segments=[
                TranscriptSegment.from_dict(segment)
                for segment in segments
                if isinstance(segment, dict)
            ],
            language=recording.transcript.language,
        )
        stats = _transcription_timing_stats(transcription)
        logger.info(
            "Speaker extraction transcript stats: segments=%d timed_segments=%d "
            "words=%d timed_words=%d min_speakers=%s max_speakers=%s",
            stats["segments"],
            stats["timed_segments"],
            stats["words"],
            stats["timed_words"],
            recording.min_speakers,
            recording.max_speakers,
        )
        if not _has_usable_timestamps(stats):
            raise ValueError(
                "Cannot extract speakers because transcript has no usable timestamps. "
                "Re-transcribe with alignment enabled."
            )

        result = transcription_service.extract_speakers(
            str(PROJECT_ROOT / recording.stored_path),
            transcription,
            model=recording.model,
            device=recording.device,
            min_speakers=recording.min_speakers,
            max_speakers=recording.max_speakers,
        )
        assigned_segments = sanitize_for_json(
            result.get("result_raw") or result.get("transcript", [])
        )
        speakers = _real_speakers(assigned_segments)
        logger.info(
            "Speaker extraction assignment stats: assigned_segments=%d speakers=%s",
            len(assigned_segments),
            sorted(speakers),
        )
        if not speakers:
            raise ValueError(
                "Speaker extraction did not assign any speakers. The transcript may "
                "not have enough timing detail. Re-transcribe with alignment enabled."
            )

        recording.transcript.segments_json = json.dumps(assigned_segments)
        recording.transcript.diarization_engine = profile.diarization_engine
        recording.transcript.model_config_json = json.dumps(
            _transcript_metadata(profile, recording)
        )
        if recording.summary is not None:
            summary = recording.summary
            recording.summary = None
            db.delete(summary)

        recording.status = RecordingStatus.DONE
        recording.error = None
        db.commit()
        logger.info("Speaker extraction worker finished recording_id=%s", recording_id)
    except Exception as exc:
        logger.exception("Speaker extraction failed: %s", exc)
        db.rollback()
        _mark_failed(db, recording_id, str(exc))
    finally:
        db.close()


def _set_status(db: Session, recording: Recording, status: str) -> None:
    recording.status = status
    recording.error = None
    db.commit()
    db.refresh(recording)


def _mark_failed(db: Session, recording_id: str, error: str) -> None:
    recording = db.get(Recording, recording_id)
    if recording is None:
        return

    recording.status = RecordingStatus.FAILED
    recording.error = error
    db.commit()


def _transcription_timing_stats(transcription: TranscriptionResult) -> dict[str, int]:
    word_count = 0
    timed_word_count = 0
    timed_segment_count = 0

    for segment in transcription.segments:
        if segment.end > segment.start:
            timed_segment_count += 1

        for word in segment.words:
            word_count += 1
            if (
                word.start is not None
                and word.end is not None
                and word.end > word.start
            ):
                timed_word_count += 1

    return {
        "segments": len(transcription.segments),
        "timed_segments": timed_segment_count,
        "words": word_count,
        "timed_words": timed_word_count,
    }


def _has_usable_timestamps(stats: dict[str, int]) -> bool:
    return stats["timed_words"] > 0 or stats["timed_segments"] > 0


def _real_speakers(segments: list[dict]) -> set[str]:
    return {
        str(segment.get("speaker")).strip()
        for segment in segments
        if isinstance(segment, dict)
        and segment.get("speaker")
        and str(segment.get("speaker")).strip().lower() != "unknown"
    }

def _transcript_metadata(
    profile,
    recording: Recording,
) -> dict:
    metadata = profile.to_metadata()
    metadata["runtime"].update({
        "language": recording.language_hint,
        "min_speakers": recording.min_speakers,
        "max_speakers": recording.max_speakers,
    })
    return metadata
