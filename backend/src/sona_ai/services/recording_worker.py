import json
import uuid

from sqlalchemy.orm import Session

from sona_ai.core import PROJECT_ROOT, sanitize_for_json, setup_logging
from sona_ai.db.engine import SessionLocal
from sona_ai.db.models import Recording, RecordingStatus, Transcript
from sona_ai.services.transcription_service import TranscriptionService


logger = setup_logging()


def run_transcription(recording_id: str, transcription_service: TranscriptionService) -> None:
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
        )

        transcript_segments = sanitize_for_json(result.get("transcript", []))
        # transcript = Transcript(
        #     id=str(uuid.uuid4()),
        #     recording_id=recording.id,
        #     segments_json=json.dumps(transcript_segments),
        #     language=recording.language_hint,
        #     transcription_engine=recording.model,
        #     diarization_engine="pyannote",
        #     model_config_json=json.dumps({
        #         "model": recording.model,
        #         "device": recording.device,
        #         "language": recording.language_hint,
        #         "min_speakers": recording.min_speakers,
        #         "max_speakers": recording.max_speakers,
        #     }),
        # )
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
