import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Optional

from sona_ai.core import setup_logging


logger = setup_logging()
RecordingJob = Callable[..., None]


@dataclass
class _JobEntry:
    recording_id: str
    cancel_event: threading.Event
    future: Optional[Future] = None


class RecordingJobManager:
    """Own background recording jobs independently from HTTP requests."""

    def __init__(self, max_workers: Optional[int] = None):
        worker_count = max_workers or _positive_int_env("SONA_RECORDING_JOB_WORKERS", 2)
        self._executor = ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="sona-recording",
        )
        self._jobs: dict[str, _JobEntry] = {}
        self._lock = threading.Lock()
        self._closed = False

    def submit(
        self,
        recording_id: str,
        job_id: str,
        target: RecordingJob,
        *args: Any,
    ) -> None:
        cancel_event = threading.Event()
        entry = _JobEntry(recording_id=recording_id, cancel_event=cancel_event)
        with self._lock:
            if self._closed:
                raise RuntimeError("Recording job manager is closed")
            if job_id in self._jobs:
                raise ValueError(f"Recording job is already registered: {job_id}")
            self._jobs[job_id] = entry

        try:
            future = self._executor.submit(
                self._run,
                job_id,
                entry,
                target,
                args,
            )
        except Exception:
            with self._lock:
                self._jobs.pop(job_id, None)
            raise

        with self._lock:
            current = self._jobs.get(job_id)
            if current is entry:
                entry.future = future

    def cancel(self, job_id: Optional[str]) -> bool:
        if not job_id:
            return False
        with self._lock:
            entry = self._jobs.get(job_id)
            if entry is None:
                return False
            entry.cancel_event.set()
            future = entry.future

        if future is not None and future.cancel():
            with self._lock:
                if self._jobs.get(job_id) is entry:
                    self._jobs.pop(job_id, None)
        return True

    def cancel_recording(self, recording_id: str) -> int:
        with self._lock:
            job_ids = [
                job_id
                for job_id, entry in self._jobs.items()
                if entry.recording_id == recording_id
            ]
        return sum(1 for job_id in job_ids if self.cancel(job_id))

    def shutdown(self, *, wait: bool = True) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            entries = list(self._jobs.values())
        for entry in entries:
            entry.cancel_event.set()
        self._executor.shutdown(wait=wait, cancel_futures=True)
        with self._lock:
            self._jobs.clear()

    def _run(
        self,
        job_id: str,
        entry: _JobEntry,
        target: RecordingJob,
        args: tuple[Any, ...],
    ) -> None:
        try:
            if entry.cancel_event.is_set():
                return
            target(*args, cancel_event=entry.cancel_event)
        except Exception:
            logger.exception(
                "Unhandled recording job failure: recording_id=%s job_id=%s",
                entry.recording_id,
                job_id,
            )
        finally:
            with self._lock:
                if self._jobs.get(job_id) is entry:
                    self._jobs.pop(job_id, None)


def _positive_int_env(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default
