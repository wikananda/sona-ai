import io
import json
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

from sona_ai.api.routes import projects


class FakeProfile:
    transcription_engine = "faster_whisper"

    def to_metadata(self):
        return {"runtime": {}}


class FakeTranscriptionService:
    def __init__(self):
        self.calls = []

    def resolve_profile(self, **kwargs):
        self.calls.append(kwargs)
        return FakeProfile()


class FakeSession:
    def __init__(self):
        self.added = None
        self.committed = False

    def get(self, model, project_id):
        return object()

    def add(self, value):
        self.added = value

    def commit(self):
        self.committed = True

    def refresh(self, recording):
        now = datetime.now(timezone.utc)
        recording.created_at = now
        recording.updated_at = now
        recording.transcript.created_at = now
        recording.transcript.updated_at = now

    def rollback(self):
        pass


class LiveRecordingSaveTest(unittest.TestCase):
    def make_request(self, service):
        return SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(transcription_service=service),
            ),
        )

    def test_saves_original_and_word_timestamps_without_alignment(self):
        service = FakeTranscriptionService()
        db = FakeSession()
        upload = SimpleNamespace(
            filename="meeting.webm",
            content_type="audio/webm",
            file=io.BytesIO(b"original audio"),
        )
        saved = SimpleNamespace(
            stored_path="data/projects/p/r.webm",
            mime_type="audio/webm",
            file_size_bytes=14,
        )
        segments = [{
            "text": "Hello",
            "start": "0.0",
            "end": "1.0",
            "words": [{
                "word": "Hello",
                "start": "0.1",
                "end": "0.8",
                "score": "0.91",
            }],
        }]

        with patch.object(projects, "save_upload", return_value=saved):
            result = projects.save_live_recording(
                "project-1",
                self.make_request(service),
                file=upload,
                segments_json=json.dumps(segments),
                language="en",
                model="faster-whisper-turbo",
                device="cpu",
                live_engine="whisper-live",
                db=db,
            )

        self.assertTrue(db.committed)
        self.assertEqual(result["stored_path"], saved.stored_path)
        self.assertEqual(result["transcript"]["segments"][0]["words"][0]["score"], 0.91)
        self.assertEqual(service.calls[0]["alignment_enabled"], False)
        runtime = result["transcript"]["model_config"]["runtime"]
        self.assertEqual(runtime["live_engine"], "whisper-live")
        self.assertTrue(runtime["word_timestamps"])

    def test_empty_live_transcript_is_valid_for_silent_recording(self):
        self.assertEqual(projects._parse_live_segments("[]"), [])

    def test_accepts_parakeet_live_engine_metadata(self):
        service = FakeTranscriptionService()
        db = FakeSession()
        upload = SimpleNamespace(
            filename="meeting.webm",
            content_type="audio/webm",
            file=io.BytesIO(b"original audio"),
        )
        saved = SimpleNamespace(
            stored_path="data/projects/p/r.webm",
            mime_type="audio/webm",
            file_size_bytes=14,
        )

        with patch.object(projects, "save_upload", return_value=saved):
            result = projects.save_live_recording(
                "project-1",
                self.make_request(service),
                file=upload,
                segments_json="[]",
                language="en",
                model="parakeet",
                device="cpu",
                live_engine="parakeet-live",
                db=db,
            )

        runtime = result["transcript"]["model_config"]["runtime"]
        self.assertEqual(runtime["live_engine"], "parakeet-live")

    def test_accepts_nemotron_live_engine_metadata(self):
        service = FakeTranscriptionService()
        db = FakeSession()
        upload = SimpleNamespace(
            filename="meeting.webm",
            content_type="audio/webm",
            file=io.BytesIO(b"original audio"),
        )
        saved = SimpleNamespace(
            stored_path="data/projects/p/r.webm",
            mime_type="audio/webm",
            file_size_bytes=14,
        )

        with patch.object(projects, "save_upload", return_value=saved):
            result = projects.save_live_recording(
                "project-1",
                self.make_request(service),
                file=upload,
                segments_json="[]",
                language="en",
                model="nemotron-3.5",
                device="cpu",
                live_engine="nemotron-live",
                db=db,
            )

        runtime = result["transcript"]["model_config"]["runtime"]
        self.assertEqual(runtime["live_engine"], "nemotron-live")

    def test_parser_clamps_invalid_word_values(self):
        segments = projects._parse_live_segments(json.dumps([{
            "text": "Safe",
            "start": "nan",
            "end": "2",
            "words": [{
                "word": "Safe",
                "start": -1,
                "end": 4,
                "score": 4,
            }],
        }]))

        self.assertEqual(segments[0]["start"], 0.0)
        self.assertEqual(segments[0]["words"][0]["start"], 0.0)
        self.assertEqual(segments[0]["words"][0]["end"], 2.0)
        self.assertEqual(segments[0]["words"][0]["score"], 1.0)


if __name__ == "__main__":
    unittest.main()
