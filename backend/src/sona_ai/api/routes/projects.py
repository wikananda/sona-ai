import json
import uuid
from typing import Optional

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from sona_ai.api.schemas.projects import ProjectCreate
from sona_ai.core import setup_logging, validate_device_available
from sona_ai.db.models import Project, Recording, RecordingStatus
from sona_ai.db.session import get_db
from sona_ai.services.recording_worker import run_transcription
from sona_ai.services.transcription_service import SUPPORTED_TRANSCRIPTION_MODELS
from sona_ai.storage import delete_project_dir, delete_recording_file, save_upload


logger = setup_logging()
router = APIRouter()


@router.post("/projects")
def create_project(body: ProjectCreate, db: Session = Depends(get_db)):
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Project name is required")

    project = Project(
        id=str(uuid.uuid4()),
        name=name,
        description=body.description.strip() if body.description else None,
    )
    db.add(project)
    db.commit()
    db.refresh(project)
    return _serialize_project(project)


@router.get("/projects")
def list_projects(db: Session = Depends(get_db)):
    projects = db.scalars(
        select(Project).order_by(Project.created_at.desc())
    ).all()
    return [_serialize_project(project) for project in projects]


@router.get("/projects/{project_id}")
def get_project(project_id: str, db: Session = Depends(get_db)):
    project = db.scalar(
        select(Project)
        .where(Project.id == project_id)
        .options(selectinload(Project.recordings))
    )
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    recordings = sorted(
        project.recordings,
        key=lambda recording: recording.created_at,
        reverse=True,
    )
    data = _serialize_project(project)
    data["recordings"] = [
        _serialize_recording(recording, include_transcript=False)
        for recording in recordings
    ]
    return data


@router.delete("/projects/{project_id}")
def delete_project(project_id: str, db: Session = Depends(get_db)):
    project = db.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    db.delete(project)
    db.commit()

    try:
        delete_project_dir(project_id)
    except Exception as exc:
        logger.warning("Failed to delete project audio directory: %s", exc)

    return {"ok": True}


@router.post("/projects/{project_id}/recordings")
def upload_project_recording(
    project_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    language: Optional[str] = Form(default=None),
    model: str = Form(default="parakeet"),
    device: str = Form(default="auto"),
    min_speakers: Optional[int] = Form(default=None),
    max_speakers: Optional[int] = Form(default=None),
    db: Session = Depends(get_db),
):
    project = db.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    model = _normalize_model(model)
    device = _normalize_device(device)
    language = _normalize_language(language)
    _validate_speakers(min_speakers, max_speakers)

    recording_id = str(uuid.uuid4())
    saved_audio = None
    try:
        saved_audio = save_upload(project_id, recording_id, file)
        recording = Recording(
            id=recording_id,
            project_id=project_id,
            original_name=file.filename or "audio",
            stored_path=saved_audio.stored_path,
            mime_type=saved_audio.mime_type,
            file_size_bytes=saved_audio.file_size_bytes,
            language_hint=language,
            model=model,
            device=device,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            status=RecordingStatus.PENDING,
        )
        db.add(recording)
        db.commit()
        db.refresh(recording)
    except Exception as exc:
        db.rollback()
        if saved_audio is not None:
            try:
                delete_recording_file(saved_audio.stored_path)
            except Exception as cleanup_exc:
                logger.warning("Failed to clean up uploaded audio: %s", cleanup_exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    background_tasks.add_task(
        run_transcription,
        recording.id,
        request.app.state.transcription_service,
    )
    return _serialize_recording(recording, include_transcript=False)


@router.get("/recordings/{recording_id}")
def get_recording(recording_id: str, db: Session = Depends(get_db)):
    recording = db.scalar(
        select(Recording)
        .where(Recording.id == recording_id)
        .options(selectinload(Recording.transcript))
    )
    if recording is None:
        raise HTTPException(status_code=404, detail="Recording not found")
    return _serialize_recording(recording, include_transcript=True)


@router.delete("/recordings/{recording_id}")
def delete_recording(recording_id: str, db: Session = Depends(get_db)):
    recording = db.get(Recording, recording_id)
    if recording is None:
        raise HTTPException(status_code=404, detail="Recording not found")

    stored_path = recording.stored_path
    db.delete(recording)
    db.commit()

    try:
        delete_recording_file(stored_path)
    except Exception as exc:
        logger.warning("Failed to delete recording audio file: %s", exc)

    return {"ok": True}


def _serialize_project(project: Project) -> dict:
    return {
        "id": project.id,
        "name": project.name,
        "description": project.description,
        "created_at": project.created_at.isoformat(),
        "updated_at": project.updated_at.isoformat(),
    }


def _serialize_recording(recording: Recording, include_transcript: bool) -> dict:
    data = {
        "id": recording.id,
        "project_id": recording.project_id,
        "original_name": recording.original_name,
        "stored_path": recording.stored_path,
        "mime_type": recording.mime_type,
        "file_size_bytes": recording.file_size_bytes,
        "language_hint": recording.language_hint,
        "model": recording.model,
        "device": recording.device,
        "min_speakers": recording.min_speakers,
        "max_speakers": recording.max_speakers,
        "status": recording.status,
        "error": recording.error,
        "created_at": recording.created_at.isoformat(),
        "updated_at": recording.updated_at.isoformat(),
    }

    if include_transcript:
        data["transcript"] = _serialize_transcript(recording)

    return data


def _serialize_transcript(recording: Recording) -> Optional[dict]:
    transcript = recording.transcript
    if transcript is None:
        return None

    return {
        "id": transcript.id,
        "recording_id": transcript.recording_id,
        "segments": json.loads(transcript.segments_json),
        "language": transcript.language,
        "transcription_engine": transcript.transcription_engine,
        "diarization_engine": transcript.diarization_engine,
        "model_config": (
            json.loads(transcript.model_config_json)
            if transcript.model_config_json
            else None
        ),
        "created_at": transcript.created_at.isoformat(),
        "updated_at": transcript.updated_at.isoformat(),
    }


def _normalize_model(model: str) -> str:
    model_name = model.lower().strip()
    if model_name not in SUPPORTED_TRANSCRIPTION_MODELS:
        allowed = ", ".join(sorted(SUPPORTED_TRANSCRIPTION_MODELS))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported transcription model. Use one of: {allowed}",
        )
    return model_name


def _normalize_device(device: str) -> str:
    try:
        return validate_device_available(device)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _normalize_language(language: Optional[str]) -> Optional[str]:
    if language is None:
        return None
    language = language.strip()
    if not language or language.lower() in {"auto", "none"}:
        return None
    return language


def _validate_speakers(
    min_speakers: Optional[int],
    max_speakers: Optional[int],
) -> None:
    if min_speakers is not None and min_speakers < 1:
        raise HTTPException(status_code=400, detail="min_speakers must be at least 1")
    if max_speakers is not None and max_speakers < 1:
        raise HTTPException(status_code=400, detail="max_speakers must be at least 1")
    if (
        min_speakers is not None
        and max_speakers is not None
        and min_speakers > max_speakers
    ):
        raise HTTPException(
            status_code=400,
            detail="min_speakers cannot be greater than max_speakers",
        )
