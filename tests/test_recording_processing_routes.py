import unittest
from types import SimpleNamespace

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from sona_ai.api.routes import projects
from sona_ai.db.engine import Base
from sona_ai.db.models import Project, Recording, RecordingStatus


class FakeJobManager:
    def __init__(self):
        self.canceled_job_ids = []

    def cancel(self, job_id):
        self.canceled_job_ids.append(job_id)
        return True


class RecordingProcessingRoutesTest(unittest.TestCase):
    def setUp(self):
        engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(engine)
        self.Session = sessionmaker(bind=engine)
        self.db = self.Session()
        self.db.add(Project(id="project-1", name="Research", description=None))
        self.db.add_all([
            _recording("queued", RecordingStatus.PENDING, "job-queued"),
            _recording("running", RecordingStatus.PROCESSING, "job-running"),
            _recording("finished", RecordingStatus.DONE, None),
        ])
        self.db.commit()

    def tearDown(self):
        self.db.close()

    def test_active_feed_contains_only_non_terminal_recordings(self):
        result = projects.list_processing_recordings(db=self.db)

        self.assertEqual([item["id"] for item in result], ["queued", "running"])
        self.assertTrue(all(item["project_name"] == "Research" for item in result))
        self.assertEqual(result[0]["progress"]["label"], "Waiting to start")
        self.assertEqual(result[1]["progress"]["label"], "Transcribing")

    def test_cancel_keeps_recording_and_notifies_managed_worker(self):
        manager = FakeJobManager()
        request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(recording_job_manager=manager),
            ),
        )

        result = projects.cancel_recording("running", request, db=self.db)

        self.assertEqual(result["status"], RecordingStatus.CANCELED)
        self.assertEqual(result["stored_path"], "data/running.wav")
        self.assertIsNotNone(self.db.get(Recording, "running"))
        self.assertEqual(manager.canceled_job_ids, ["job-running"])


def _recording(recording_id: str, status: str, job_id: str | None) -> Recording:
    return Recording(
        id=recording_id,
        project_id="project-1",
        original_name=f"{recording_id}.wav",
        stored_path=f"data/{recording_id}.wav",
        mime_type="audio/wav",
        file_size_bytes=100,
        language_hint="en",
        model="parakeet",
        device="cpu",
        status=status,
        processing_stage=(
            "queued"
            if status == RecordingStatus.PENDING
            else "transcribing"
            if status == RecordingStatus.PROCESSING
            else "done"
        ),
        processing_job_id=job_id,
        progress_completed_steps=0,
        progress_total_steps=1,
    )


if __name__ == "__main__":
    unittest.main()
