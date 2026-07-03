import json
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from sona_ai.api.routes._errors import route_error_handler
from sona_ai.api.routes._project_helpers import (
    _serialize_recording,
    _summary_text_from_segments,
)
from sona_ai.api.schemas.projects import RecordingSummaryUpdate
from sona_ai.api.schemas.summarize import RecordingSummaryRequest
from sona_ai.db.models import Recording, RecordingSummary
from sona_ai.db.session import get_db


router = APIRouter()


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

    with route_error_handler("Error summarizing recording: %s", byok_errors=True):
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


@router.patch("/recordings/{recording_id}/summary")
def update_recording_summary(
    recording_id: str,
    body: RecordingSummaryUpdate,
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
    if recording.summary is None:
        raise HTTPException(status_code=404, detail="Summary not found")

    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Summary text cannot be empty")

    recording.summary.text = text
    recording.summary.strategy = "manual_edit"
    db.commit()
    db.refresh(recording)
    db.refresh(recording.summary)
    return _serialize_recording(recording, include_transcript=True)
