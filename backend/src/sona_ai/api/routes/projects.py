import json
import httpx
import uuid
from typing import Optional

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Body,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from sona_ai.api.schemas.projects import (
    ProjectCreate,
    RecordingRename,
    RecordingRetranscribe,
    RecordingSpeakerExtraction,
    TranscriptSegmentUpdate,
    TranscriptSpeakerRename,
)
from sona_ai.api.schemas.summarize import RecordingSummaryRequest
from sona_ai.core import PROJECT_ROOT, setup_logging, validate_device_available
from sona_ai.db.models import Project, Recording, RecordingStatus, RecordingSummary
from sona_ai.db.session import get_db
from sona_ai.services.recording_worker import run_speaker_extraction, run_transcription
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
    extract_speakers: bool = Form(default=True),
    db: Session = Depends(get_db),
):
    project = db.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    model = _normalize_model(model)
    device = _normalize_device(device)
    language = _normalize_language(language)
    _validate_speaker_settings(extract_speakers, min_speakers, max_speakers)
    profile = request.app.state.transcription_service.resolve_profile(
        model=model,
        device=device,
        extract_speakers=extract_speakers,
    )
    progress_total_steps = _transcription_step_count(profile)

    recording_id = str(uuid.uuid4())
    job_id = str(uuid.uuid4())
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
            processing_stage="queued",
            processing_job_id=job_id,
            progress_completed_steps=0,
            progress_total_steps=progress_total_steps,
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
        job_id,
        request.app.state.transcription_service,
        extract_speakers,
    )
    return _serialize_recording(recording, include_transcript=False)

@router.patch("/recordings/{recording_id}")
def rename_recording(
    recording_id: str,
    body: RecordingRename,
    db: Session = Depends(get_db),
):
    recording = db.get(Recording, recording_id)
    if recording is None:
        raise HTTPException(status_code=404, detail="Recording not found")

    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=404, detail="Recording name is required")

    recording.original_name = name
    db.commit()
    db.refresh(recording)

    return _serialize_recording(recording, include_transcript=True)

@router.get("/recordings/{recording_id}")
def get_recording(recording_id: str, db: Session = Depends(get_db)):
    recording = db.scalar(
        select(Recording)
        .where(Recording.id == recording_id)
        .options(
            selectinload(Recording.transcript),
            selectinload(Recording.summary),
        )
    )
    if recording is None:
        raise HTTPException(status_code=404, detail="Recording not found")
    return _serialize_recording(recording, include_transcript=True)


@router.post("/recordings/{recording_id}/summary")
async def summarize_recording(
    recording_id: str,
    request: Request,
    body: RecordingSummaryRequest,
    db: Session = Depends(get_db),
):
    recording = db.scalar(
        select(Recording)
        .where(Recording.id == recording_id)
        .options(
            selectinload(Recording.transcript),
            selectinload(Recording.summary),
        )
    )
    if recording is None:
        raise HTTPException(status_code=404, detail="Recording not found")
    if recording.transcript is None:
        raise HTTPException(status_code=404, detail="Transcript not found")

    transcript_text = _summary_text_from_segments(
        json.loads(recording.transcript.segments_json)
    )
    if not transcript_text:
        raise HTTPException(status_code=400, detail="Transcript is empty")

    try:
        result = await run_in_threadpool(
            request.app.state.summarization_service.summarize_adaptive,
            transcript_text,
            body.prompt,
            max_length=body.max_length,
            model=body.model,
            device=body.device,
            mode=body.mode,
            byok=body.byok.model_dump() if body.byok else None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        if status == 401:
            raise HTTPException(
                status_code=400,
                detail="BYOK authentication failed. Check provider, API key, and model.",
            ) from exc
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Error summarizing recording: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    summary = recording.summary
    if summary is None:
        summary = RecordingSummary(
            id=str(uuid.uuid4()),
            recording_id=recording.id,
        )
        recording.summary = summary
        db.add(summary)

    summary.text = result["summary"]
    summary.format_name = result["format_name"]
    summary.plan_json = json.dumps(result["plan"]) if result["plan"] else None
    summary.strategy = "adaptive"
    summary.mode = body.mode
    summary.model = body.model if body.mode == "local" else None
    summary.device = body.device if body.mode == "local" else None
    summary.provider = body.byok.provider if body.mode == "byok" and body.byok else None
    summary.provider_model = body.byok.model if body.mode == "byok" and body.byok else None

    db.commit()
    db.refresh(recording)
    db.refresh(summary)
    return _serialize_recording(recording, include_transcript=True)


@router.get("/recordings/{recording_id}/audio")
def get_recording_audio(recording_id: str, db: Session = Depends(get_db)):
    recording = db.get(Recording, recording_id)
    if recording is None:
        raise HTTPException(status_code=404, detail="Recording not found")

    audio_path = (PROJECT_ROOT / recording.stored_path).resolve()
    if not audio_path.is_file():
        raise HTTPException(status_code=404, detail="Recording audio file not found")

    return FileResponse(
        audio_path,
        media_type=recording.mime_type or "application/octet-stream",
        filename=recording.original_name,
    )


@router.post("/recordings/{recording_id}/retranscribe")
def retranscribe_recording(
    recording_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
    body: RecordingRetranscribe | None = Body(default=None),
    db: Session = Depends(get_db),
):
    recording = db.get(Recording, recording_id)
    if recording is None:
        raise HTTPException(status_code=404, detail="Recording not found")
    if recording.status in {RecordingStatus.PENDING, RecordingStatus.PROCESSING}:
        raise HTTPException(
            status_code=409,
            detail="Recording transcription is already running",
        )
    if not (PROJECT_ROOT / recording.stored_path).is_file():
        raise HTTPException(
            status_code=404,
            detail=(
                "Recording audio file is missing. Re-upload the audio before "
                "running transcription again."
            ),
        )

    extract_speakers = body.extract_speakers if body is not None else True
    if body is not None:
        recording.language_hint = _normalize_language(body.language)
        recording.model = _normalize_model(body.model or recording.model)
        recording.device = _normalize_device(body.device or recording.device)
        recording.min_speakers = body.min_speakers
        recording.max_speakers = body.max_speakers
    _validate_speaker_settings(
        extract_speakers,
        recording.min_speakers,
        recording.max_speakers,
    )
    profile = request.app.state.transcription_service.resolve_profile(
        model=recording.model,
        device=recording.device,
        extract_speakers=extract_speakers,
    )
    job_id = str(uuid.uuid4())

    recording.status = RecordingStatus.PENDING
    recording.processing_stage = "queued"
    recording.processing_job_id = job_id
    recording.progress_completed_steps = 0
    recording.progress_total_steps = _transcription_step_count(profile)
    recording.error = None
    if recording.summary is not None:
        summary = recording.summary
        recording.summary = None
        db.delete(summary)
    db.commit()
    db.refresh(recording)

    background_tasks.add_task(
        run_transcription,
        recording.id,
        job_id,
        request.app.state.transcription_service,
        extract_speakers,
    )
    return _serialize_recording(recording, include_transcript=False)


@router.post("/recordings/{recording_id}/speakers/extract")
def extract_recording_speakers(
    recording_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
    body: RecordingSpeakerExtraction | None = Body(default=None),
    db: Session = Depends(get_db),
):
    recording = db.scalar(
        select(Recording)
        .where(Recording.id == recording_id)
        .options(selectinload(Recording.transcript))
    )
    if recording is None:
        raise HTTPException(status_code=404, detail="Recording not found")
    if recording.status in {RecordingStatus.PENDING, RecordingStatus.PROCESSING}:
        raise HTTPException(
            status_code=409,
            detail="Recording processing is already running",
        )
    if recording.transcript is None:
        raise HTTPException(status_code=404, detail="Transcript not found")
    if recording.transcript.diarization_engine:
        raise HTTPException(status_code=409, detail="Speakers have already been extracted")
    if not (PROJECT_ROOT / recording.stored_path).is_file():
        raise HTTPException(
            status_code=404,
            detail="Recording audio file is missing. Re-upload the audio before extracting speakers.",
        )

    if body is not None:
        recording.min_speakers = body.min_speakers
        recording.max_speakers = body.max_speakers
    _validate_speaker_settings(
        True,
        recording.min_speakers,
        recording.max_speakers,
    )
    job_id = str(uuid.uuid4())

    recording.status = RecordingStatus.PENDING
    recording.processing_stage = "queued"
    recording.processing_job_id = job_id
    recording.progress_completed_steps = 0
    recording.progress_total_steps = 2
    recording.error = None
    db.commit()
    db.refresh(recording)

    background_tasks.add_task(
        run_speaker_extraction,
        recording.id,
        job_id,
        request.app.state.transcription_service,
    )
    return _serialize_recording(recording, include_transcript=True)


@router.post("/recordings/{recording_id}/cancel")
def cancel_recording(recording_id: str, db: Session = Depends(get_db)):
    recording = db.scalar(
        select(Recording)
        .where(Recording.id == recording_id)
        .options(
            selectinload(Recording.transcript),
            selectinload(Recording.summary),
        )
    )
    if recording is None:
        raise HTTPException(status_code=404, detail="Recording not found")
    if recording.status not in {RecordingStatus.PENDING, RecordingStatus.PROCESSING}:
        raise HTTPException(
            status_code=409,
            detail="Recording is not currently processing",
        )

    recording.status = RecordingStatus.CANCELED
    recording.processing_stage = "canceled"
    recording.processing_job_id = None
    recording.error = None
    db.commit()
    db.refresh(recording)
    return _serialize_recording(recording, include_transcript=True)


@router.patch("/recordings/{recording_id}/transcript/speakers")
def rename_transcript_speakers(
    recording_id: str,
    body: TranscriptSpeakerRename,
    db: Session = Depends(get_db),
):
    recording = db.scalar(
        select(Recording)
        .where(Recording.id == recording_id)
        .options(selectinload(Recording.transcript))
    )
    if recording is None:
        raise HTTPException(status_code=404, detail="Recording not found")
    if recording.transcript is None:
        raise HTTPException(status_code=404, detail="Transcript not found")

    speaker_names = {
        speaker: name.strip()
        for speaker, name in body.speakers.items()
    }
    if any(not name for name in speaker_names.values()):
        raise HTTPException(status_code=400, detail="Speaker names cannot be empty")

    segments = json.loads(recording.transcript.segments_json)
    present_speakers = {
        segment.get("speaker")
        for segment in segments
        if isinstance(segment, dict) and segment.get("speaker")
    }
    speaker_names = {
        speaker: name
        for speaker, name in speaker_names.items()
        if speaker in present_speakers
    }

    if speaker_names:
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            speaker = segment.get("speaker")
            if speaker in speaker_names:
                segment["speaker"] = speaker_names[speaker]

        recording.transcript.segments_json = json.dumps(segments)
        if recording.summary is not None:
            summary = recording.summary
            recording.summary = None
            db.delete(summary)
        db.commit()
        db.refresh(recording)
        db.refresh(recording.transcript)

    return _serialize_recording(recording, include_transcript=True)


@router.patch("/recordings/{recording_id}/transcript/segments/{segment_index}")
def update_transcript_segment(
    recording_id: str,
    segment_index: int,
    body: TranscriptSegmentUpdate,
    db: Session = Depends(get_db),
):
    recording = db.scalar(
        select(Recording)
        .where(Recording.id == recording_id)
        .options(
            selectinload(Recording.transcript),
            selectinload(Recording.summary),
        )
    )
    if recording is None:
        raise HTTPException(status_code=404, detail="Recording not found")
    if recording.transcript is None:
        raise HTTPException(status_code=404, detail="Transcript not found")

    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Transcript text cannot be empty")

    segments = json.loads(recording.transcript.segments_json)
    if segment_index < 0 or segment_index >= len(segments):
        raise HTTPException(status_code=404, detail="Transcript segment not found")

    segment = segments[segment_index]
    if not isinstance(segment, dict):
        raise HTTPException(status_code=400, detail="Transcript segment is invalid")

    segment["text"] = text
    if segment.get("speaker") is not None:
        speaker = (body.speaker or "").strip()
        if not speaker:
            raise HTTPException(status_code=400, detail="Speaker name cannot be empty")
        segment["speaker"] = speaker

    recording.transcript.segments_json = json.dumps(segments)
    if recording.summary is not None:
        summary = recording.summary
        recording.summary = None
        db.delete(summary)

    db.commit()
    db.refresh(recording)
    db.refresh(recording.transcript)
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


def _serialize_progress(recording: Recording) -> dict:
    total_steps = max(int(recording.progress_total_steps or 0), 0)
    completed_steps = max(int(recording.progress_completed_steps or 0), 0)
    if total_steps > 0:
        completed_steps = min(completed_steps, total_steps)
        percent = round((completed_steps / total_steps) * 100)
    else:
        percent = 100 if recording.status == RecordingStatus.DONE else 0

    stage = recording.processing_stage or _stage_for_status(recording.status)
    return {
        "stage": stage,
        "label": _progress_label(stage),
        "completed_steps": completed_steps,
        "total_steps": total_steps,
        "percent": percent,
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
        "progress": _serialize_progress(recording),
        "error": recording.error,
        "created_at": recording.created_at.isoformat(),
        "updated_at": recording.updated_at.isoformat(),
    }

    if include_transcript:
        data["transcript"] = _serialize_transcript(recording)
        data["summary"] = _serialize_summary(recording)

    return data


def _stage_for_status(status: str) -> str:
    if status == RecordingStatus.DONE:
        return "done"
    if status == RecordingStatus.FAILED:
        return "failed"
    if status == RecordingStatus.CANCELED:
        return "canceled"
    if status == RecordingStatus.PENDING:
        return "queued"
    return "processing"


def _progress_label(stage: str) -> str:
    return {
        "queued": "Waiting to start",
        "preparing": "Preparing models",
        "transcribing": "Transcribing",
        "aligning": "Aligning",
        "diarizing": "Diarizing",
        "assigning_speakers": "Assigning speakers",
        "done": "Done",
        "failed": "Failed",
        "canceled": "Canceled",
        "processing": "Processing",
    }.get(stage, "Processing")


def _transcription_step_count(profile) -> int:
    total_steps = 1
    if profile.alignment_enabled:
        total_steps += 1
    if profile.diarization_enabled:
        total_steps += 2
    return total_steps


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


def _serialize_summary(recording: Recording) -> Optional[dict]:
    summary = recording.summary
    if summary is None:
        return None

    return {
        "id": summary.id,
        "recording_id": summary.recording_id,
        "text": summary.text,
        "mode": summary.mode,
        "model": summary.model,
        "device": summary.device,
        "provider": summary.provider,
        "provider_model": summary.provider_model,
        "format_name": summary.format_name,
        "plan": json.loads(summary.plan_json) if summary.plan_json else None,
        "strategy": summary.strategy,
        "created_at": summary.created_at.isoformat(),
        "updated_at": summary.updated_at.isoformat(),
    }


def _summary_text_from_segments(segments: list) -> str:
    lines = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue

        text = str(segment.get("text") or "").strip()
        if not text:
            continue

        speaker = str(segment.get("speaker") or "").strip()
        if speaker:
            lines.append(f"{speaker}: {text}")
        else:
            lines.append(text)

    return "\n".join(lines)


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


def _validate_speaker_settings(
    extract_speakers: bool,
    min_speakers: Optional[int],
    max_speakers: Optional[int],
) -> None:
    if extract_speakers and (min_speakers is None or max_speakers is None):
        raise HTTPException(
            status_code=400,
            detail="min_speakers and max_speakers are required when extracting speakers",
        )
    _validate_speakers(min_speakers, max_speakers)
