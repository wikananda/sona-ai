import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from fastapi import UploadFile

from sona_ai.core import PROJECT_ROOT


PROJECT_AUDIO_ROOT = PROJECT_ROOT / "data" / "projects"


@dataclass(frozen=True)
class SavedAudio:
    stored_path: str
    mime_type: Optional[str]
    file_size_bytes: int


def save_upload(project_id: str, recording_id: str, upload_file: UploadFile) -> SavedAudio:
    extension = Path(upload_file.filename or "").suffix.lower()
    if not extension:
        extension = ".audio"

    project_dir = _safe_project_dir(project_id)
    project_dir.mkdir(parents=True, exist_ok=True)

    destination = project_dir / f"{recording_id}{extension}"
    size = 0
    with destination.open("wb") as buffer:
        while chunk := upload_file.file.read(1024 * 1024):
            size += len(chunk)
            buffer.write(chunk)

    return SavedAudio(
        stored_path=str(destination.relative_to(PROJECT_ROOT)),
        mime_type=upload_file.content_type,
        file_size_bytes=size,
    )


def delete_recording_file(stored_path: str) -> None:
    path = _safe_project_path(stored_path)
    if path.exists() and path.is_file():
        path.unlink()


def delete_project_dir(project_id: str) -> None:
    project_dir = _safe_project_dir(project_id)
    if project_dir.exists():
        shutil.rmtree(project_dir)


def _safe_project_dir(project_id: str) -> Path:
    path = PROJECT_AUDIO_ROOT / project_id
    resolved = path.resolve()
    root = PROJECT_AUDIO_ROOT.resolve()
    if root != resolved and root not in resolved.parents:
        raise ValueError("Invalid project path")
    return resolved


def _safe_project_path(stored_path: str) -> Path:
    path = (PROJECT_ROOT / stored_path).resolve()
    root = PROJECT_AUDIO_ROOT.resolve()
    if root != path and root not in path.parents:
        raise ValueError("Invalid recording path")
    return path
