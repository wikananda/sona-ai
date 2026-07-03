import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from sona_ai.api.routes._project_helpers import _serialize_recording
from sona_ai.api.schemas.projects import (
    TranscriptSegmentUpdate,
    TranscriptSpeakerRename,
)
from sona_ai.db.models import Recording
from sona_ai.db.session import get_db


router = APIRouter()


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
