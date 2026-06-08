import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sona_ai.storage import audio


class AudioStorageTest(unittest.TestCase):
    def test_save_upload_preserves_original_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir).resolve()
            project_audio_root = project_root / "data" / "projects"

            upload = SimpleNamespace(
                filename="meeting.webm",
                content_type="audio/webm;codecs=opus",
                file=io.BytesIO(b"raw audio"),
            )

            with (
                patch.object(audio, "PROJECT_ROOT", project_root),
                patch.object(audio, "PROJECT_AUDIO_ROOT", project_audio_root),
            ):
                saved = audio.save_upload("project-1", "recording-1", upload)

            self.assertEqual(saved.stored_path, "data/projects/project-1/recording-1.webm")
            self.assertEqual(saved.mime_type, "audio/webm;codecs=opus")
            self.assertEqual(saved.file_size_bytes, len(b"raw audio"))
            self.assertTrue((project_root / saved.stored_path).is_file())
            self.assertEqual((project_root / saved.stored_path).read_bytes(), b"raw audio")

    def test_save_upload_cleans_partial_original_when_write_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir).resolve()
            project_audio_root = project_root / "data" / "projects"
            paths = {}

            def fake_write(destination: Path, upload_file) -> int:
                paths["original"] = destination
                destination.write_bytes(b"partial original")
                raise RuntimeError("write failed")

            upload = SimpleNamespace(
                filename="meeting.webm",
                content_type="audio/webm;codecs=opus",
                file=io.BytesIO(b"raw audio"),
            )

            with (
                patch.object(audio, "PROJECT_ROOT", project_root),
                patch.object(audio, "PROJECT_AUDIO_ROOT", project_audio_root),
                patch.object(audio, "_write_upload", side_effect=fake_write),
            ):
                with self.assertRaisesRegex(RuntimeError, "write failed"):
                    audio.save_upload("project-1", "recording-1", upload)

            self.assertFalse(paths["original"].exists())

    def test_ensure_transcription_audio_creates_asr_copy_and_keeps_source(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir).resolve()
            project_audio_root = project_root / "data" / "projects"
            source = project_audio_root / "project-1" / "recording-1.webm"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"raw audio")

            def fake_convert(input_path: Path, output_path: Path) -> None:
                self.assertEqual(input_path, source)
                output_path.write_bytes(b"normalized wav")

            with (
                patch.object(audio, "PROJECT_ROOT", project_root),
                patch.object(audio, "PROJECT_AUDIO_ROOT", project_audio_root),
                patch.object(audio, "_convert_to_wav", side_effect=fake_convert),
            ):
                saved = audio.ensure_transcription_audio(
                    "data/projects/project-1/recording-1.webm"
                )

            self.assertEqual(saved.stored_path, "data/projects/project-1/recording-1.asr.wav")
            self.assertEqual(saved.mime_type, "audio/wav")
            self.assertEqual(saved.file_size_bytes, len(b"normalized wav"))
            self.assertTrue(source.exists())
            self.assertTrue((project_root / saved.stored_path).is_file())

    def test_ensure_transcription_audio_keeps_source_when_conversion_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir).resolve()
            project_audio_root = project_root / "data" / "projects"
            source = project_audio_root / "project-1" / "recording-1.webm"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"raw audio")
            normalized = source.with_name("recording-1.asr.wav")

            def fake_convert(input_path: Path, output_path: Path) -> None:
                output_path.write_bytes(b"partial wav")
                raise RuntimeError("conversion failed")

            with (
                patch.object(audio, "PROJECT_ROOT", project_root),
                patch.object(audio, "PROJECT_AUDIO_ROOT", project_audio_root),
                patch.object(audio, "_convert_to_wav", side_effect=fake_convert),
            ):
                with self.assertRaisesRegex(RuntimeError, "conversion failed"):
                    audio.ensure_transcription_audio(
                        "data/projects/project-1/recording-1.webm"
                    )

            self.assertTrue(source.is_file())
            self.assertFalse(normalized.exists())

    def test_delete_recording_file_removes_original_and_asr_copy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir).resolve()
            project_audio_root = project_root / "data" / "projects"
            source = project_audio_root / "project-1" / "recording-1.webm"
            normalized = project_audio_root / "project-1" / "recording-1.asr.wav"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"raw audio")
            normalized.write_bytes(b"normalized wav")

            with (
                patch.object(audio, "PROJECT_ROOT", project_root),
                patch.object(audio, "PROJECT_AUDIO_ROOT", project_audio_root),
            ):
                audio.delete_recording_file("data/projects/project-1/recording-1.webm")

            self.assertFalse(source.exists())
            self.assertFalse(normalized.exists())


if __name__ == "__main__":
    unittest.main()
