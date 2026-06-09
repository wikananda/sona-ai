import json
import os
import subprocess
import threading
import traceback
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal, Optional
from uuid import uuid4

from dotenv import load_dotenv

from sona_ai.core import (
    PROJECT_ROOT,
    load_config,
    model_cache_root,
    model_manifest_dir,
    setup_logging,
    setup_model_cache_environment,
)
from sona_ai.services.pipeline_profile import PipelineProfile


logger = setup_logging()
ModelStatus = Literal["missing", "installed", "running", "failed"]
JobStatus = Literal["queued", "running", "installed", "failed"]


@dataclass(frozen=True)
class ModelCatalogEntry:
    id: str
    label: str
    type: str
    model_names: list[str]
    config_name: str
    environment: str
    cache_subdir: Optional[str] = None
    requires_hf_token: bool = False


@dataclass
class DownloadJob:
    job_id: str
    model_id: str
    status: JobStatus
    message: str
    started_at: str
    finished_at: Optional[str] = None
    error: Optional[str] = None


class ModelDownloadService:
    def __init__(self):
        self._jobs: dict[str, DownloadJob] = {}
        self._lock = threading.Lock()

    def list_models(self) -> list[dict]:
        active_jobs = self._active_jobs_by_model()
        models = []
        for entry in MODEL_CATALOG:
            active_job = active_jobs.get(entry.id)
            installed = self._manifest_path(entry.id).is_file()
            status: ModelStatus = "installed" if installed else "missing"
            error = None
            if active_job:
                status = "running" if active_job.status in {"queued", "running"} else active_job.status
                error = active_job.error

            models.append(
                {
                    **asdict(entry),
                    "installed": installed,
                    "status": status,
                    "cache_path": self._display_cache_path(self._cache_path(entry)),
                    "requires_hf_token": entry.requires_hf_token,
                    "hf_token_available": self._hf_token_available(),
                    "active_job_id": active_job.job_id if active_job else None,
                    "error": error,
                }
            )
        return models

    def start_download(self, model_id: str) -> DownloadJob:
        entry = self._entry(model_id)
        if self._manifest_path(model_id).is_file():
            return self._create_finished_job(
                model_id=model_id,
                status="installed",
                message=f"{entry.label} is already installed.",
            )

        with self._lock:
            for job in self._jobs.values():
                if job.model_id == model_id and job.status in {"queued", "running"}:
                    return job

            job = DownloadJob(
                job_id=str(uuid4()),
                model_id=model_id,
                status="queued",
                message=f"Queued {entry.label} download.",
                started_at=self._now(),
            )
            self._jobs[job.job_id] = job

        thread = threading.Thread(target=self._run_download, args=(entry, job.job_id), daemon=True)
        thread.start()
        return job

    def get_job(self, job_id: str) -> DownloadJob:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            return self._jobs[job_id]

    def required_model_ids_for_profile(self, profile: PipelineProfile) -> list[str]:
        model_ids: list[str] = []

        transcription_model_id = self._transcription_model_id(profile)
        if transcription_model_id is not None:
            model_ids.append(transcription_model_id)

        if profile.alignment_enabled and profile.alignment_engine in {"wav2vec2", "wav2vec2_external"}:
            model_ids.append("wav2vec2-aligner")

        if profile.diarization_enabled and profile.diarization_engine == "community_external":
            model_ids.append("pyannote-community")

        return model_ids

    def required_models_for_profile(self, profile: PipelineProfile) -> list[dict]:
        by_id = {model["id"]: model for model in self.list_models()}
        return [
            by_id[model_id]
            for model_id in self.required_model_ids_for_profile(profile)
            if model_id in by_id
        ]

    def mark_installed(self, model_id: str) -> None:
        entry = self._entry(model_id)
        self._write_manifest(entry)

    def mark_profile_installed(self, profile: PipelineProfile) -> None:
        for model_id in self.required_model_ids_for_profile(profile):
            self.mark_installed(model_id)

    def _run_download(self, entry: ModelCatalogEntry, job_id: str) -> None:
        self._update_job(job_id, status="running", message=f"Downloading {entry.label}...")
        try:
            load_dotenv(PROJECT_ROOT / ".env")
            if entry.requires_hf_token and not self._hf_token_available():
                raise EnvironmentError("HF_TOKEN is required for this model.")

            downloader = DOWNLOADERS[entry.id]
            downloader(entry)
            self._write_manifest(entry)
            self._update_job(
                job_id,
                status="installed",
                message=f"{entry.label} is installed.",
                finished_at=self._now(),
            )
        except Exception as exc:
            logger.error("Model download failed for %s: %s", entry.id, exc)
            logger.debug(traceback.format_exc())
            self._update_job(
                job_id,
                status="failed",
                message=f"{entry.label} download failed.",
                finished_at=self._now(),
                error=str(exc),
            )

    def _create_finished_job(self, model_id: str, status: JobStatus, message: str) -> DownloadJob:
        job = DownloadJob(
            job_id=str(uuid4()),
            model_id=model_id,
            status=status,
            message=message,
            started_at=self._now(),
            finished_at=self._now(),
        )
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def _update_job(self, job_id: str, **patch) -> None:
        with self._lock:
            job = self._jobs[job_id]
            for key, value in patch.items():
                setattr(job, key, value)

    def _active_jobs_by_model(self) -> dict[str, DownloadJob]:
        active: dict[str, DownloadJob] = {}
        with self._lock:
            for job in self._jobs.values():
                if job.status in {"queued", "running", "failed"}:
                    active[job.model_id] = job
        return active

    def _entry(self, model_id: str) -> ModelCatalogEntry:
        for entry in MODEL_CATALOG:
            if entry.id == model_id:
                return entry
        raise KeyError(model_id)

    def _transcription_model_id(self, profile: PipelineProfile) -> Optional[str]:
        if profile.transcription_config == "parakeet":
            return "parakeet"
        if profile.transcription_config == "faster-whisper-large-v3":
            return "faster-whisper-large-v3"
        if profile.transcription_config == "faster-whisper-turbo":
            return "faster-whisper-turbo"
        return None

    def _cache_path(self, entry: ModelCatalogEntry) -> Path:
        root = model_cache_root(load_config(entry.config_name))
        return root / entry.cache_subdir if entry.cache_subdir else root

    def _display_cache_path(self, path: Path) -> str:
        try:
            return str(path.relative_to(PROJECT_ROOT))
        except ValueError:
            return str(path)

    def _manifest_path(self, model_id: str) -> Path:
        return model_manifest_dir() / f"{model_id}.json"

    def _write_manifest(self, entry: ModelCatalogEntry) -> None:
        manifest_dir = model_manifest_dir()
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "id": entry.id,
            "label": entry.label,
            "type": entry.type,
            "model_names": entry.model_names,
            "cache_path": str(self._cache_path(entry)),
            "status": "installed",
            "downloaded_at": self._now(),
        }
        self._manifest_path(entry.id).write_text(json.dumps(manifest, indent=2))

    def _hf_token_available(self) -> bool:
        load_dotenv(PROJECT_ROOT / ".env")
        return bool(os.getenv("HF_TOKEN"))

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()


def _download_parakeet(entry: ModelCatalogEntry) -> None:
    from sona_ai.transcription.parakeet_transcriber import ParakeetTranscriber

    config = deepcopy(load_config(entry.config_name))
    config.setdefault("model", {})["device"] = "cpu"
    transcriber = ParakeetTranscriber(config)
    try:
        transcriber.load_models()
    finally:
        transcriber.cleanup_models()


def _download_faster_whisper(entry: ModelCatalogEntry) -> None:
    from sona_ai.transcription.faster_whisper_transcriber import FasterWhisperTranscriber

    config = deepcopy(load_config(entry.config_name))
    config.setdefault("model", {})["device"] = "cpu"
    transcriber = FasterWhisperTranscriber(config)
    try:
        transcriber.load_models()
    finally:
        transcriber.cleanup_models()


def _download_wav2vec2(entry: ModelCatalogEntry) -> None:
    config = load_config(entry.config_name)
    setup_model_cache_environment(config)
    cache_dir = model_cache_root(config) / "wav2vec2-align"
    script_path = PROJECT_ROOT / "tools" / "alignment" / "download_wav2vec2_models.py"
    cmd = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        "sona-aligner",
        "python",
        str(script_path),
        "--cache-dir",
        str(cache_dir),
    ]
    for model_name in entry.model_names:
        cmd.extend(["--model-name", model_name])
    _run_command(cmd)


def _download_pyannote(entry: ModelCatalogEntry) -> None:
    config = load_config(entry.config_name)
    setup_model_cache_environment(config)
    cache_dir = model_cache_root(config) / "pyannote-community"
    script_path = PROJECT_ROOT / "tools" / "diarization" / "download_community_model.py"
    cmd = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        "sona-diarization",
        "python",
        str(script_path),
        "--cache-dir",
        str(cache_dir),
    ]
    _run_command(cmd)


def _run_command(cmd: list[str]) -> None:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT, env=env)


MODEL_CATALOG = [
    ModelCatalogEntry(
        id="parakeet",
        label="Parakeet 0.6B v3",
        type="transcription",
        model_names=["nvidia/parakeet-tdt-0.6b-v3"],
        config_name="parakeet",
        environment="sona-ai",
    ),
    ModelCatalogEntry(
        id="faster-whisper-large-v3",
        label="Faster-Whisper large-v3",
        type="transcription",
        model_names=["large-v3"],
        config_name="faster-whisper-large-v3",
        environment="sona-ai",
        cache_subdir="faster-whisper",
    ),
    ModelCatalogEntry(
        id="faster-whisper-turbo",
        label="Faster-Whisper turbo",
        type="transcription",
        model_names=["turbo"],
        config_name="faster-whisper-turbo",
        environment="sona-ai",
        cache_subdir="faster-whisper",
    ),
    ModelCatalogEntry(
        id="wav2vec2-aligner",
        label="Wav2Vec2 aligner",
        type="alignment",
        model_names=[
            "facebook/wav2vec2-base-960h",
            "indonesian-nlp/wav2vec2-large-xlsr-indonesian",
        ],
        config_name="wav2vec2",
        environment="sona-aligner",
        cache_subdir="wav2vec2-align",
    ),
    ModelCatalogEntry(
        id="pyannote-community",
        label="pyannote Community-1",
        type="diarization",
        model_names=["pyannote/speaker-diarization-community-1"],
        config_name="diarization-community",
        environment="sona-diarization",
        cache_subdir="pyannote-community",
        requires_hf_token=True,
    ),
]

DOWNLOADERS: dict[str, Callable[[ModelCatalogEntry], None]] = {
    "parakeet": _download_parakeet,
    "faster-whisper-large-v3": _download_faster_whisper,
    "faster-whisper-turbo": _download_faster_whisper,
    "wav2vec2-aligner": _download_wav2vec2,
    "pyannote-community": _download_pyannote,
}

model_download_service = ModelDownloadService()
