import shutil
import subprocess
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
    project_dir = _safe_project_dir(project_id)
    project_dir.mkdir(parents=True, exist_ok=True)

    destination = project_dir / f"{recording_id}.wav"
    size = save_upload_as_wav(upload_file, destination)

    return SavedAudio(
        stored_path=str(destination.relative_to(PROJECT_ROOT)),
        mime_type="audio/wav",
        file_size_bytes=size,
    )


def save_upload_as_wav(upload_file: UploadFile, destination: Path) -> int:
    extension = Path(upload_file.filename or "").suffix.lower()
    if not extension:
        extension = ".audio"

    destination.parent.mkdir(parents=True, exist_ok=True)
    raw_destination = destination.with_name(f"{destination.stem}.upload{extension}")

    try:
        _write_upload(raw_destination, upload_file)
        _convert_to_wav(raw_destination, destination)
        return destination.stat().st_size
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        raw_destination.unlink(missing_ok=True)


def normalize_recording_file(stored_path: str) -> SavedAudio:
    source = _safe_project_path(stored_path)
    if not source.is_file():
        raise FileNotFoundError(f"Recording audio file not found: {stored_path}")

    if source.suffix.lower() == ".wav":
        return SavedAudio(
            stored_path=str(source.relative_to(PROJECT_ROOT)),
            mime_type="audio/wav",
            file_size_bytes=source.stat().st_size,
        )

    destination = source.with_suffix(".wav")
    try:
        _convert_to_wav(source, destination)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    else:
        source.unlink(missing_ok=True)

    return SavedAudio(
        stored_path=str(destination.relative_to(PROJECT_ROOT)),
        mime_type="audio/wav",
        file_size_bytes=destination.stat().st_size,
    )


def _write_upload(destination: Path, upload_file: UploadFile) -> int:
    size = 0
    with destination.open("wb") as buffer:
        while chunk := upload_file.file.read(1024 * 1024):
            size += len(chunk)
            buffer.write(chunk)
    return size


def _convert_to_wav(input_path: Path, output_path: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg is required to normalize uploaded audio before transcription."
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        detail = f": {stderr}" if stderr else ""
        raise RuntimeError(f"Failed to normalize uploaded audio with ffmpeg{detail}") from exc


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
