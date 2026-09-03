import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from sona_ai.db.engine import Base
from sona_ai.db.models import Project, Recording, RecordingStatus
from sona_ai.services import recording_worker


class FakeProfile:
    alignment_enabled = False
    diarization_enabled = False
    transcription_engine = "fake"


class CancelDuringTranscriptionService:
    def __init__(self, session_factory, cancel_event):
        self.session_factory = session_factory
        self.cancel_event = cancel_event

    def resolve_profile(self, **kwargs):
        return FakeProfile()

    def transcribe(self, *args, **kwargs):
        # Simulate another API request canceling while native inference is running.
        with self.session_factory() as db:
            recording = db.get(Recording, "recording-1")
            recording.status = RecordingStatus.CANCELED
            recording.processing_stage = "canceled"
            recording.processing_job_id = None
            db.commit()
        self.cancel_event.set()
        return {
            "result_raw": [{"text": "late result", "start": 0.0, "end": 1.0}],
        }


class RecordingWorkerCancellationTest(unittest.TestCase):
    def test_late_transcription_result_is_not_saved_after_cancel(self):
        engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        with Session() as db:
            db.add(Project(id="project-1", name="Project", description=None))
            db.add(Recording(
                id="recording-1",
                project_id="project-1",
                original_name="original.wav",
                stored_path="data/original.wav",
                mime_type="audio/wav",
                file_size_bytes=100,
                language_hint="en",
                model="parakeet",
                device="cpu",
                status=RecordingStatus.PENDING,
                processing_stage="queued",
                processing_job_id="job-1",
                progress_completed_steps=0,
                progress_total_steps=1,
            ))
            db.commit()

        cancel_event = threading.Event()
        service = CancelDuringTranscriptionService(Session, cancel_event)
        with (
            patch.object(recording_worker, "SessionLocal", Session),
            patch.object(
                recording_worker,
                "_ensure_transcription_audio",
                return_value=SimpleNamespace(stored_path="data/original.wav"),
            ),
        ):
            recording_worker.run_transcription(
                "recording-1",
                "job-1",
                service,
                extract_speakers=False,
                cancel_event=cancel_event,
            )

        with Session() as db:
            recording = db.get(Recording, "recording-1")
            self.assertEqual(recording.status, RecordingStatus.CANCELED)
            self.assertEqual(recording.stored_path, "data/original.wav")
            self.assertIsNone(recording.transcript)


if __name__ == "__main__":
    unittest.main()
