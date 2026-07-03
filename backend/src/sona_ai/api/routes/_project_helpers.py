import json
from typing import Optional

from fastapi import HTTPException

from sona_ai.core import sanitize_for_json, validate_device_available
from sona_ai.db.models import Project, Recording, RecordingStatus
from sona_ai.services.transcription_service import SUPPORTED_TRANSCRIPTION_MODELS


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


def _offset_segments(segments: list[dict], offset_seconds: float) -> list[dict]:
    offset_segments = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        shifted = dict(segment)
        shifted["start"] = float(shifted.get("start") or 0.0) + offset_seconds
        shifted["end"] = float(shifted.get("end") or 0.0) + offset_seconds
        words = []
        for word in shifted.get("words") or []:
            if not isinstance(word, dict):
                continue
            shifted_word = dict(word)
            if shifted_word.get("start") is not None:
                shifted_word["start"] = float(shifted_word["start"]) + offset_seconds
            if shifted_word.get("end") is not None:
                shifted_word["end"] = float(shifted_word["end"]) + offset_seconds
            words.append(shifted_word)
        if words:
            shifted["words"] = words
        offset_segments.append(shifted)
    return sanitize_for_json(offset_segments)


def _has_usable_segment_timestamps(segments: list[dict]) -> bool:
    return any(
        float(segment.get("end") or 0.0) > float(segment.get("start") or 0.0)
        for segment in segments
        if isinstance(segment, dict)
    )


def _parse_live_segments(segments_json: str) -> list[dict]:
    try:
        segments = json.loads(segments_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="segments_json must be valid JSON") from exc

    if not isinstance(segments, list):
        raise HTTPException(status_code=400, detail="segments_json must be a list")

    parsed_segments = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        text = str(segment.get("text") or "").strip()
        if not text:
            continue
        start = float(segment.get("start") or 0.0)
        end = float(segment.get("end") or start)
        parsed_segments.append(
            {
                "text": text,
                "start": start,
                "end": max(end, start),
            }
        )

    if not parsed_segments:
        raise HTTPException(status_code=400, detail="Live transcript has no text segments")

    return sanitize_for_json(parsed_segments)


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
