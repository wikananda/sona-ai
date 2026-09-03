import json
import os
import shutil
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
JobStatus = Literal["queued", "running", "installed", "uninstalled", "failed"]
JobAction = Literal["download", "uninstall", "redownload"]
JobStage = Literal["queued", "preparing", "downloading", "removing", "verifying", "done", "failed"]


@dataclass(frozen=True)
class ModelCatalogEntry:
    id: str
    label: str
    type: str
    model_names: list[str]
    config_name: str
    environment: str
    runtime_cache_subdir: Optional[str] = None
    requires_hf_token: bool = False
    can_uninstall: bool = True
    can_redownload: bool = True
    unsupported_reason: Optional[str] = None
    management_note: Optional[str] = None


@dataclass
class ModelJob:
    job_id: str
    model_id: str
    action: JobAction
    status: JobStatus
    stage: JobStage
    message: str
    indeterminate: bool
    started_at: str
    finished_at: Optional[str] = None
    error: Optional[str] = None


class ModelDownloadService:
    def __init__(self):
        self._jobs: dict[str, ModelJob] = {}
        self._lock = threading.Lock()

    def list_models(self) -> list[dict]:
        latest_jobs = self._latest_jobs_by_model()
        models = []
        for entry in MODEL_CATALOG:
            latest_job = latest_jobs.get(entry.id)
            installed = self._manifest_path(entry.id).is_file()
            status: ModelStatus = "installed" if installed else "missing"
            error = None
            if latest_job:
                if latest_job.status in {"queued", "running"}:
                    status = "running"
                elif latest_job.status == "failed":
                    status = "failed"
                error = latest_job.error

            models.append(
                {
                    **asdict(entry),
                    "installed": installed,
                    "status": status,
                    "cache_path": self._display_cache_path(self._cache_path(entry)),
                    "requires_hf_token": entry.requires_hf_token,
                    "hf_token_available": self._hf_token_available(),
                    "active_job_id": (
                        latest_job.job_id
                        if latest_job and latest_job.status in {"queued", "running", "failed"}
                        else None
                    ),
                    "is_busy": bool(latest_job and latest_job.status in {"queued", "running"}),
                    "error": error,
                }
            )
        return models

    def start_download(self, model_id: str) -> ModelJob:
        entry = self._entry(model_id)
        if self._manifest_path(model_id).is_file():
            return self._create_finished_job(
                model_id=model_id,
                action="download",
                status="installed",
                stage="done",
                message=f"{entry.label} is already installed.",
            )

        return self._start_job(entry, action="download")

    def start_uninstall(self, model_id: str) -> ModelJob:
        entry = self._entry(model_id)
        if not entry.can_uninstall:
            raise ValueError(entry.unsupported_reason or f"{entry.label} cannot be uninstalled safely.")
        if not self._manifest_path(model_id).is_file():
            return self._create_finished_job(
                model_id=model_id,
                action="uninstall",
                status="uninstalled",
                stage="done",
                message=f"{entry.label} is already removed.",
            )

        return self._start_job(entry, action="uninstall")

    def start_redownload(self, model_id: str) -> ModelJob:
        entry = self._entry(model_id)
        if not entry.can_redownload:
            raise ValueError(entry.unsupported_reason or f"{entry.label} cannot be re-downloaded safely.")
        return self._start_job(entry, action="redownload")

    def _start_job(self, entry: ModelCatalogEntry, action: JobAction) -> ModelJob:
        with self._lock:
            for job in self._jobs.values():
                if job.model_id == entry.id and job.status in {"queued", "running"}:
                    return job

            verb = self._action_label(action)
            job = ModelJob(
                job_id=str(uuid4()),
                model_id=entry.id,
                action=action,
                status="queued",
                stage="queued",
                message=f"Queued {entry.label} {verb}.",
                indeterminate=True,
                started_at=self._now(),
            )
            self._jobs[job.job_id] = job

        thread = threading.Thread(target=self._run_job, args=(entry, job.job_id), daemon=True)
        thread.start()
        return job

    def get_job(self, job_id: str) -> ModelJob:
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
            entry = self._entry(model_id)
            if entry.environment == "sona-ai":
                self.mark_installed(model_id)

    def _run_job(self, entry: ModelCatalogEntry, job_id: str) -> None:
        job = self.get_job(job_id)
        try:
            load_dotenv(PROJECT_ROOT / ".env")
            if job.action in {"download", "redownload"} and entry.requires_hf_token and not self._hf_token_available():
                raise EnvironmentError("HF_TOKEN is required for this model.")

            if job.action == "download":
                self._run_download(entry, job_id)
            elif job.action == "uninstall":
                self._run_uninstall(entry, job_id)
            elif job.action == "redownload":
                self._run_redownload(entry, job_id)
            else:
                raise ValueError(f"Unsupported model job action: {job.action}")
        except Exception as exc:
            logger.error("Model job failed for %s (%s): %s", entry.id, job.action, exc)
            logger.debug(traceback.format_exc())
            self._update_job(
                job_id,
                status="failed",
                stage="failed",
                message=f"{entry.label} {self._action_label(job.action)} failed.",
                finished_at=self._now(),
                error=str(exc),
            )

    def _run_download(self, entry: ModelCatalogEntry, job_id: str) -> None:
        self._update_job(
            job_id,
            status="running",
            stage="preparing",
            message=f"Preparing {entry.label} download...",
        )
        downloader = DOWNLOADERS[entry.id]
        self._update_job(
            job_id,
            status="running",
            stage="downloading",
            message=f"Downloading {entry.label}...",
        )
        downloader(entry)
        self._update_job(
            job_id,
            status="running",
            stage="verifying",
            message=f"Verifying {entry.label} files...",
        )
        self._write_manifest(entry)
        self._update_job(
            job_id,
            status="installed",
            stage="done",
            message=f"{entry.label} is installed.",
            finished_at=self._now(),
        )

    def _run_uninstall(self, entry: ModelCatalogEntry, job_id: str) -> None:
        self._update_job(
            job_id,
            status="running",
            stage="removing",
            message=f"Removing {entry.label}...",
        )
        self._remove_managed_files(entry)
        self._update_job(
            job_id,
            status="running",
            stage="verifying",
            message=f"Verifying {entry.label} removal...",
        )
        self._clear_install_state(entry)
        self._update_job(
            job_id,
            status="uninstalled",
            stage="done",
            message=f"{entry.label} was removed.",
            finished_at=self._now(),
        )

    def _run_redownload(self, entry: ModelCatalogEntry, job_id: str) -> None:
        self._update_job(
            job_id,
            status="running",
            stage="removing",
            message=f"Removing previous {entry.label} files...",
        )
        self._remove_managed_files(entry)
        self._clear_install_state(entry)
        self._update_job(
            job_id,
            status="running",
            stage="preparing",
            message=f"Preparing fresh {entry.label} download...",
        )
        downloader = DOWNLOADERS[entry.id]
        self._update_job(
            job_id,
            status="running",
            stage="downloading",
            message=f"Downloading {entry.label} again...",
        )
        downloader(entry)
        self._update_job(
            job_id,
            status="running",
            stage="verifying",
            message=f"Verifying fresh {entry.label} files...",
        )
        self._write_manifest(entry)
        self._update_job(
            job_id,
            status="installed",
            stage="done",
            message=f"{entry.label} is installed.",
            finished_at=self._now(),
        )

    def _create_finished_job(
        self,
        model_id: str,
        action: JobAction,
        status: JobStatus,
        stage: JobStage,
        message: str,
    ) -> ModelJob:
        job = ModelJob(
            job_id=str(uuid4()),
            model_id=model_id,
            action=action,
            status=status,
            stage=stage,
            message=message,
            indeterminate=True,
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

    def _latest_jobs_by_model(self) -> dict[str, ModelJob]:
        latest: dict[str, ModelJob] = {}
        with self._lock:
            for job in self._jobs.values():
                latest[job.model_id] = job
        return latest

    def _entry(self, model_id: str) -> ModelCatalogEntry:
        for entry in MODEL_CATALOG:
            if entry.id == model_id:
                return entry
        raise KeyError(model_id)

    def _transcription_model_id(self, profile: PipelineProfile) -> Optional[str]:
        if profile.transcription_config == "parakeet":
            return "parakeet"
        if profile.transcription_config == "nemotron-3.5":
            return "nemotron-3.5"
        if profile.transcription_config == "faster-whisper-large-v3":
            return "faster-whisper-large-v3"
        if profile.transcription_config == "faster-whisper-turbo":
            return "faster-whisper-turbo"
        return None

    def _cache_path(self, entry: ModelCatalogEntry) -> Path:
        return model_cache_root(model_id=entry.id)

    def _display_cache_path(self, path: Path) -> str:
        try:
            return str(path.relative_to(PROJECT_ROOT))
        except ValueError:
            return str(path)

    def _manifest_path(self, model_id: str) -> Path:
        return model_manifest_dir() / f"{model_id}.json"

    def _clear_manifest(self, model_id: str) -> None:
        manifest_path = self._manifest_path(model_id)
        if manifest_path.is_file():
            manifest_path.unlink()

    def _clear_install_state(self, entry: ModelCatalogEntry) -> None:
        self._clear_manifest(entry.id)

    def _remove_managed_files(self, entry: ModelCatalogEntry) -> None:
        targets = self._managed_paths(entry)
        for path in targets:
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()

    def _managed_paths(self, entry: ModelCatalogEntry) -> list[Path]:
        return [self._cache_path(entry)]

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
            "management_note": entry.management_note,
        }
        self._manifest_path(entry.id).write_text(json.dumps(manifest, indent=2))

    def _hf_token_available(self) -> bool:
        load_dotenv(PROJECT_ROOT / ".env")
        return bool(os.getenv("HF_TOKEN"))

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _action_label(self, action: JobAction) -> str:
        if action == "download":
            return "download"
        if action == "uninstall":
            return "removal"
        return "re-download"


def _download_parakeet(entry: ModelCatalogEntry) -> None:
    from sona_ai.transcription.parakeet_transcriber import ParakeetTranscriber

    config = deepcopy(load_config(entry.config_name))
    config["_sona_managed_model_id"] = entry.id
    config.setdefault("model", {})["device"] = "cpu"
    transcriber = ParakeetTranscriber(config)
    try:
        transcriber.load_models()
    finally:
        transcriber.cleanup_models()


def _download_faster_whisper(entry: ModelCatalogEntry) -> None:
    from sona_ai.transcription.faster_whisper_transcriber import FasterWhisperTranscriber

    config = deepcopy(load_config(entry.config_name))
    config["_sona_managed_model_id"] = entry.id
    config.setdefault("model", {})["device"] = "cpu"
    transcriber = FasterWhisperTranscriber(config)
    try:
        transcriber.load_models()
    finally:
        transcriber.cleanup_models()


def _download_nemotron(entry: ModelCatalogEntry) -> None:
    from huggingface_hub import hf_hub_download

    from sona_ai.transcription.nemotron_transcriber import NEMOTRON_GGUF_FILENAME

    cache_dir = model_cache_root(model_id=entry.id)
    cache_dir.mkdir(parents=True, exist_ok=True)
    downloaded_path = Path(hf_hub_download(
        repo_id=entry.model_names[0],
        filename=NEMOTRON_GGUF_FILENAME,
        local_dir=str(cache_dir),
    ))
    if not downloaded_path.is_file() or downloaded_path.stat().st_size <= 0:
        raise RuntimeError("Nemotron 3.5 model download did not produce a usable GGUF file.")


def _download_wav2vec2(entry: ModelCatalogEntry) -> None:
    config = load_config(entry.config_name)
    config["_sona_managed_model_id"] = entry.id
    setup_model_cache_environment(config)
    cache_dir = model_cache_root(config) / (entry.runtime_cache_subdir or "wav2vec2-align")
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
    config["_sona_managed_model_id"] = entry.id
    setup_model_cache_environment(config)
    cache_dir = model_cache_root(config) / (entry.runtime_cache_subdir or "pyannote-community")
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
        id="nemotron-3.5",
        label="Nemotron 3.5 ASR 0.6B",
        type="transcription",
        model_names=["nvidia/nemotron-3.5-asr-streaming-0.6b"],
        config_name="nemotron-3.5",
        environment="nemotron-sidecar",
        management_note=(
            "Downloads the q8 GGUF into .models/nemotron-3.5 for the isolated "
            "NeMo-Speech.cpp server."
        ),
    ),
    ModelCatalogEntry(
        id="parakeet",
        label="Parakeet 0.6B v3",
        type="transcription",
        model_names=["nvidia/parakeet-tdt-0.6b-v3"],
        config_name="parakeet",
        environment="sona-ai",
        management_note="Uses an isolated NeMo/Hugging Face cache root under .models/parakeet.",
    ),
    ModelCatalogEntry(
        id="faster-whisper-large-v3",
        label="Faster-Whisper large-v3",
        type="transcription",
        model_names=["large-v3"],
        config_name="faster-whisper-large-v3",
        environment="sona-ai",
        runtime_cache_subdir="faster-whisper",
    ),
    ModelCatalogEntry(
        id="faster-whisper-turbo",
        label="Faster-Whisper turbo",
        type="transcription",
        model_names=["turbo"],
        config_name="faster-whisper-turbo",
        environment="sona-ai",
        runtime_cache_subdir="faster-whisper",
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
        runtime_cache_subdir="wav2vec2-align",
    ),
    ModelCatalogEntry(
        id="pyannote-community",
        label="pyannote Community-1",
        type="diarization",
        model_names=["pyannote/speaker-diarization-community-1"],
        config_name="diarization-community",
        environment="sona-diarization",
        runtime_cache_subdir="pyannote-community",
        requires_hf_token=True,
    ),
]

DOWNLOADERS: dict[str, Callable[[ModelCatalogEntry], None]] = {
    "nemotron-3.5": _download_nemotron,
    "parakeet": _download_parakeet,
    "faster-whisper-large-v3": _download_faster_whisper,
    "faster-whisper-turbo": _download_faster_whisper,
    "wav2vec2-aligner": _download_wav2vec2,
    "pyannote-community": _download_pyannote,
}

model_download_service = ModelDownloadService()
