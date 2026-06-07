import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sona_ai.storage import audio


class AudioStorageTest(unittest.TestCase):
    def test_save_upload_normalizes_to_wav_and_deletes_raw_upload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir).resolve()
            project_audio_root = project_root / "data" / "projects"
            raw_paths = []

            def fake_convert(input_path: Path, output_path: Path) -> None:
                raw_paths.append(input_path)
                self.assertEqual(input_path.read_bytes(), b"raw audio")
                output_path.write_bytes(b"normalized wav")

            upload = SimpleNamespace(
                filename="meeting.webm",
                content_type="audio/webm;codecs=opus",
                file=io.BytesIO(b"raw audio"),
            )

            with (
                patch.object(audio, "PROJECT_ROOT", project_root),
                patch.object(audio, "PROJECT_AUDIO_ROOT", project_audio_root),
                patch.object(audio, "_convert_to_wav", side_effect=fake_convert),
            ):
                saved = audio.save_upload("project-1", "recording-1", upload)

            self.assertEqual(saved.stored_path, "data/projects/project-1/recording-1.wav")
            self.assertEqual(saved.mime_type, "audio/wav")
            self.assertEqual(saved.file_size_bytes, len(b"normalized wav"))
            self.assertTrue((project_root / saved.stored_path).is_file())
            self.assertEqual(len(raw_paths), 1)
            self.assertFalse(raw_paths[0].exists())

    def test_save_upload_cleans_partial_files_when_normalization_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir).resolve()
            project_audio_root = project_root / "data" / "projects"
            paths = {}

            def fake_convert(input_path: Path, output_path: Path) -> None:
                paths["raw"] = input_path
                paths["normalized"] = output_path
                output_path.write_bytes(b"partial wav")
                raise RuntimeError("conversion failed")

            upload = SimpleNamespace(
                filename="meeting.webm",
                content_type="audio/webm;codecs=opus",
                file=io.BytesIO(b"raw audio"),
            )

            with (
                patch.object(audio, "PROJECT_ROOT", project_root),
                patch.object(audio, "PROJECT_AUDIO_ROOT", project_audio_root),
                patch.object(audio, "_convert_to_wav", side_effect=fake_convert),
            ):
                with self.assertRaisesRegex(RuntimeError, "conversion failed"):
                    audio.save_upload("project-1", "recording-1", upload)

            self.assertFalse(paths["raw"].exists())
            self.assertFalse(paths["normalized"].exists())

    def test_normalize_recording_file_converts_existing_non_wav_file(self):
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
                saved = audio.normalize_recording_file(
                    "data/projects/project-1/recording-1.webm"
                )

            self.assertEqual(saved.stored_path, "data/projects/project-1/recording-1.wav")
            self.assertEqual(saved.mime_type, "audio/wav")
            self.assertEqual(saved.file_size_bytes, len(b"normalized wav"))
            self.assertFalse(source.exists())
            self.assertTrue((project_root / saved.stored_path).is_file())

    def test_normalize_recording_file_keeps_source_when_conversion_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir).resolve()
            project_audio_root = project_root / "data" / "projects"
            source = project_audio_root / "project-1" / "recording-1.webm"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"raw audio")
            normalized = source.with_suffix(".wav")

            def fake_convert(input_path: Path, output_path: Path) -> None:
                output_path.write_bytes(b"partial wav")
                raise RuntimeError("conversion failed")

            with (
                patch.object(audio, "PROJECT_ROOT", project_root),
                patch.object(audio, "PROJECT_AUDIO_ROOT", project_audio_root),
                patch.object(audio, "_convert_to_wav", side_effect=fake_convert),
            ):
                with self.assertRaisesRegex(RuntimeError, "conversion failed"):
                    audio.normalize_recording_file(
                        "data/projects/project-1/recording-1.webm"
                    )

            self.assertTrue(source.is_file())
            self.assertFalse(normalized.exists())


if __name__ == "__main__":
    unittest.main()
