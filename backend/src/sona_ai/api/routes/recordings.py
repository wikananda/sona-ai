import json
import tempfile
import uuid
from pathlib import Path
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

from sona_ai.api.routes._project_helpers import (
    _has_usable_segment_timestamps,
    _normalize_device,
    _normalize_language,
    _normalize_model,
    _offset_segments,
    _parse_live_segments,
    _serialize_recording,
    _transcription_step_count,
    _validate_speaker_settings,
)
from sona_ai.api.schemas.projects import (
    RecordingRename,
    RecordingRetranscribe,
    RecordingSpeakerExtraction,
)
from sona_ai.core import PROJECT_ROOT, sanitize_for_json, setup_logging
from sona_ai.db.models import Project, Recording, RecordingStatus, Transcript
from sona_ai.db.session import get_db
from sona_ai.services.recording_worker import run_speaker_extraction, run_transcription
from sona_ai.storage import (
    delete_recording_file,
    ensure_transcription_audio,
    save_upload,
    save_upload_as_wav,
)


logger = setup_logging()
router = APIRouter()


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


@router.post("/projects/{project_id}/live-transcription/chunks")
async def transcribe_live_chunk(
    project_id: str,
    request: Request,
    file: UploadFile = File(...),
    chunk_index: int = Form(...),
    chunk_start: float = Form(default=0.0),
    language: Optional[str] = Form(default=None),
    model: str = Form(default="parakeet"),
    device: str = Form(default="auto"),
    db: Session = Depends(get_db),
):
    if db.get(Project, project_id) is None:
        raise HTTPException(status_code=404, detail="Project not found")

    model = _normalize_model(model)
    device = _normalize_device(device)
    language = _normalize_language(language)
    if chunk_index < 0:
        raise HTTPException(status_code=400, detail="chunk_index must be at least 0")
    if chunk_start < 0:
        raise HTTPException(status_code=400, detail="chunk_start must be at least 0")

    with tempfile.TemporaryDirectory() as temp_dir:
        audio_path = Path(temp_dir) / f"live-chunk-{chunk_index}.wav"
        try:
            await run_in_threadpool(save_upload_as_wav, file, audio_path)
            transcription = await run_in_threadpool(
                request.app.state.transcription_service.transcribe_live_chunk,
                str(audio_path),
                language=language,
                model=model,
                device=device,
            )
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    segments = _offset_segments(transcription.to_segment_dicts(), chunk_start)
    return {
        "chunk_index": chunk_index,
        "chunk_start": chunk_start,
        "segments": segments,
        "language": transcription.language or language,
    }


@router.post("/projects/{project_id}/live-transcription/recordings")
def save_live_recording(
    project_id: str,
    request: Request,
    file: UploadFile = File(...),
    segments_json: str = Form(...),
    language: Optional[str] = Form(default=None),
    model: str = Form(default="parakeet"),
    device: str = Form(default="auto"),
    db: Session = Depends(get_db),
):
    if db.get(Project, project_id) is None:
        raise HTTPException(status_code=404, detail="Project not found")

    model = _normalize_model(model)
    device = _normalize_device(device)
    language = _normalize_language(language)
    segments = _parse_live_segments(segments_json)
    profile = request.app.state.transcription_service.resolve_profile(
        model=model,
        device=device,
        alignment_enabled=True,
        extract_speakers=False,
    )

    recording_id = str(uuid.uuid4())
    saved_audio = None
    try:
        saved_audio = save_upload(project_id, recording_id, file)
        recording = Recording(
            id=recording_id,
            project_id=project_id,
            original_name=file.filename or "live-recording.webm",
            stored_path=saved_audio.stored_path,
            mime_type=saved_audio.mime_type,
            file_size_bytes=saved_audio.file_size_bytes,
            language_hint=language,
            model=model,
            device=device,
            status=RecordingStatus.DONE,
            processing_stage="done",
            processing_job_id=None,
            progress_completed_steps=1,
            progress_total_steps=1,
            error=None,
        )
        alignment_used = False
        alignment_error = None
        transcription_audio = ensure_transcription_audio(saved_audio.stored_path)
        final_segments = segments
        try:
            aligned_segments, transcript_changed = (
                request.app.state.transcription_service.align_live_segments(
                    str(PROJECT_ROOT / transcription_audio.stored_path),
                    language,
                    segments,
                    model=model,
                    device=device,
                )
            )
            final_segments = aligned_segments
            alignment_used = (
                transcript_changed
                and _has_usable_segment_timestamps(final_segments)
            )
        except Exception as exc:
            alignment_error = str(exc)
            logger.warning(
                "Failed to align live recording %s; saving rough live timestamps: %s",
                recording_id,
                exc,
            )

        transcript_metadata = profile.to_metadata()
        transcript_metadata["runtime"]["language"] = language
        transcript_metadata["runtime"]["live_transcription"] = True
        transcript_metadata["runtime"]["live_alignment_used"] = alignment_used
        if alignment_error:
            transcript_metadata["runtime"]["live_alignment_error"] = alignment_error
        transcript = Transcript(
            id=str(uuid.uuid4()),
            recording_id=recording_id,
            segments_json=json.dumps(sanitize_for_json(final_segments)),
            language=language,
            transcription_engine=profile.transcription_engine,
            diarization_engine=None,
            model_config_json=json.dumps(transcript_metadata),
        )
        recording.transcript = transcript
        db.add(recording)
        db.commit()
        db.refresh(recording)
    except Exception as exc:
        db.rollback()
        if saved_audio is not None:
            try:
                delete_recording_file(saved_audio.stored_path)
            except Exception as cleanup_exc:
                logger.warning("Failed to clean up live recording audio: %s", cleanup_exc)
        if isinstance(exc, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return _serialize_recording(recording, include_transcript=True)

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
