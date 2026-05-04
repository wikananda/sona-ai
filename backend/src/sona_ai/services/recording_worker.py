import json
import uuid

from sqlalchemy.orm import Session

from sona_ai.core import PROJECT_ROOT, sanitize_for_json, setup_logging
from sona_ai.db.engine import SessionLocal
from sona_ai.db.models import Recording, RecordingStatus, Transcript
from sona_ai.services.transcription_service import TranscriptionService


logger = setup_logging()


def run_transcription(recording_id: str, transcription_service: TranscriptionService) -> None:
    db = SessionLocal()
    try:
        recording = db.get(Recording, recording_id)
        if recording is None:
            return

        _set_status(db, recording, RecordingStatus.PROCESSING)

        result = transcription_service.transcribe(
            str(PROJECT_ROOT / recording.stored_path),
            language=recording.language_hint,
            model=recording.model,
            min_speakers=recording.min_speakers,
            max_speakers=recording.max_speakers,
        )

        transcript_segments = sanitize_for_json(result.get("transcript", []))
        transcript = Transcript(
            id=str(uuid.uuid4()),
            recording_id=recording.id,
            segments_json=json.dumps(transcript_segments),
            language=recording.language_hint,
            transcription_engine=recording.model,
            diarization_engine="pyannote",
            model_config_json=json.dumps({
                "model": recording.model,
                "language": recording.language_hint,
                "min_speakers": recording.min_speakers,
                "max_speakers": recording.max_speakers,
            }),
        )

        if recording.transcript is not None:
            db.delete(recording.transcript)
            db.flush()

        db.add(transcript)
        recording.status = RecordingStatus.DONE
        recording.error = None
        db.commit()
    except Exception as exc:
        logger.exception("Recording transcription failed: %s", exc)
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
