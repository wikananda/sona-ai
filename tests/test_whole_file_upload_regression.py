import io
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException

from sona_ai.api.routes import projects
from sona_ai.services.recording_worker import run_transcription


class FakeProfile:
    alignment_enabled = True
    diarization_enabled = False


class FakeTranscriptionService:
    def resolve_profile(self, **kwargs):
        return FakeProfile()


class FakeSession:
    def __init__(self):
        self.recording = None

    def get(self, model, project_id):
        return object()

    def add(self, recording):
        self.recording = recording

    def commit(self):
        pass

    def refresh(self, recording):
        now = datetime.now(timezone.utc)
        recording.created_at = now
        recording.updated_at = now

    def rollback(self):
        pass


class FakeBackgroundTasks:
    def __init__(self):
        self.calls = []

    def add_task(self, function, *args, **kwargs):
        self.calls.append((function, args, kwargs))


class WholeFileUploadRegressionTest(unittest.TestCase):
    def test_upload_still_saves_original_and_enqueues_normal_worker(self):
        service = FakeTranscriptionService()
        request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(transcription_service=service),
            ),
        )
        upload = SimpleNamespace(
            filename="whole-meeting.m4a",
            content_type="audio/mp4",
            file=io.BytesIO(b"complete original recording"),
        )
        saved = SimpleNamespace(
            stored_path="data/projects/project-1/recording.m4a",
            mime_type="audio/mp4",
            file_size_bytes=27,
        )
        background_tasks = FakeBackgroundTasks()
        db = FakeSession()

        with patch.object(projects, "save_upload", return_value=saved) as save_upload:
            result = projects.upload_project_recording(
                "project-1",
                request,
                background_tasks,
                file=upload,
                language="auto",
                model="faster-whisper-turbo",
                device="cpu",
                min_speakers=None,
                max_speakers=None,
                extract_speakers=False,
                db=db,
            )

        save_upload.assert_called_once()
        self.assertEqual(result["stored_path"], saved.stored_path)
        self.assertEqual(result["status"], "pending")
        self.assertEqual(len(background_tasks.calls), 1)
        worker, args, kwargs = background_tasks.calls[0]
        self.assertIs(worker, run_transcription)
        self.assertEqual(args[0], result["id"])
        self.assertFalse(args[-1])
        self.assertEqual(kwargs, {})

    def test_nemotron_upload_preserves_original_and_uses_normal_worker(self):
        service = FakeTranscriptionService()
        request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(transcription_service=service),
            ),
        )
        upload = SimpleNamespace(
            filename="whole-meeting.webm",
            content_type="audio/webm",
            file=io.BytesIO(b"complete original recording"),
        )
        saved = SimpleNamespace(
            stored_path="data/projects/project-1/recording.webm",
            mime_type="audio/webm",
            file_size_bytes=27,
        )
        background_tasks = FakeBackgroundTasks()
        db = FakeSession()

        with patch.object(projects, "save_upload", return_value=saved) as save_upload:
            result = projects.upload_project_recording(
                "project-1",
                request,
                background_tasks,
                file=upload,
                language="en",
                model="nemotron-3.5",
                device="cpu",
                min_speakers=None,
                max_speakers=None,
                extract_speakers=False,
                db=db,
            )

        save_upload.assert_called_once()
        self.assertEqual(result["stored_path"], saved.stored_path)
        self.assertEqual(result["model"], "nemotron-3.5")
        self.assertEqual(background_tasks.calls[0][0], run_transcription)

    def test_nemotron_upload_rejects_indonesian_before_saving(self):
        with self.assertRaisesRegex(HTTPException, "does not support"):
            projects._validate_model_language("nemotron-3.5", "id")


if __name__ == "__main__":
    unittest.main()
