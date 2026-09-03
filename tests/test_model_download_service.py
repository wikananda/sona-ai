import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from sona_ai.core import (
    model_cache_root,
    model_manifest_dir,
    setup_model_cache_environment,
)
from sona_ai.services.model_download_service import (
    _download_nemotron,
    _download_whisper_mps,
    model_download_service,
)
from sona_ai.services.pipeline_profile import PipelineProfile


class ModelDownloadServiceTest(unittest.TestCase):
    def wait_for_job(self, job_id: str, timeout: float = 5.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            job = model_download_service.get_job(job_id)
            if job.status in {"installed", "uninstalled", "failed"}:
                return job
            time.sleep(0.05)
        self.fail(f"Model job {job_id} did not finish within {timeout} seconds.")

    def test_required_model_ids_include_transcription_alignment_and_diarization(self):
        profile = PipelineProfile(
            transcription_engine="faster_whisper",
            transcription_config="faster-whisper-large-v3",
            alignment_enabled=True,
            alignment_engine="wav2vec2_external",
            alignment_config="wav2vec2",
            diarization_enabled=True,
            diarization_engine="community_external",
            diarization_config="diarization-community",
            device="cpu",
        )

        self.assertEqual(
            model_download_service.required_model_ids_for_profile(profile),
            [
                "faster-whisper-large-v3",
                "wav2vec2-aligner",
                "pyannote-community",
            ],
        )

    def test_mark_profile_installed_writes_manifests(self):
        profile = PipelineProfile(
            transcription_engine="parakeet",
            transcription_config="parakeet",
            alignment_enabled=False,
            alignment_engine="none",
            alignment_config=None,
            diarization_enabled=False,
            diarization_engine="community_external",
            diarization_config="diarization-community",
            device="cpu",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(os.environ, {"SONA_HF_CACHE": temp_dir}, clear=False):
                model_download_service.mark_profile_installed(profile)

                manifest_path = model_manifest_dir() / "parakeet.json"
                self.assertTrue(manifest_path.is_file())

                manifest = json.loads(manifest_path.read_text())
                self.assertEqual(manifest["id"], "parakeet")
                self.assertEqual(
                    Path(manifest["cache_path"]).resolve(),
                    (Path(temp_dir) / "parakeet").resolve(),
                )

    def test_list_models_exposes_management_capabilities(self):
        models = {
            model["id"]: model
            for model in model_download_service.list_models()
        }

        self.assertTrue(models["parakeet"]["can_uninstall"])
        self.assertTrue(models["parakeet"]["can_redownload"])
        self.assertTrue(models["faster-whisper-large-v3"]["can_uninstall"])
        self.assertTrue(models["wav2vec2-aligner"]["can_redownload"])
        self.assertTrue(models["parakeet"]["cache_path"].endswith(".models/parakeet"))
        self.assertEqual(models["nemotron-3.5"]["environment"], "nemotron-sidecar")
        self.assertTrue(models["nemotron-3.5"]["cache_path"].endswith(".models/nemotron-3.5"))

    def test_nemotron_profile_requires_managed_gguf(self):
        profile = PipelineProfile(
            transcription_engine="nemotron",
            transcription_config="nemotron-3.5",
            alignment_enabled=False,
            alignment_engine="none",
            alignment_config=None,
            diarization_enabled=False,
            diarization_engine="community_external",
            diarization_config="diarization-community",
            device="cpu",
        )

        self.assertEqual(
            model_download_service.required_model_ids_for_profile(profile),
            ["nemotron-3.5"],
        )

    def test_mps_profile_requires_distinct_transformers_weights(self):
        profile = PipelineProfile(
            transcription_engine="whisper_mps",
            transcription_config="whisper-mps-turbo",
            alignment_enabled=False,
            alignment_engine="none",
            alignment_config=None,
            diarization_enabled=False,
            diarization_engine="community_external",
            diarization_config="diarization-community",
            device="mps",
        )

        self.assertEqual(
            model_download_service.required_model_ids_for_profile(profile),
            ["whisper-mps-turbo"],
        )

    def test_mps_downloader_uses_pinned_transformers_revision(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(os.environ, {"SONA_HF_CACHE": temp_dir}, clear=False):
                entry = model_download_service._entry("whisper-mps-turbo")

                with patch("huggingface_hub.snapshot_download") as download:
                    _download_whisper_mps(entry)

                download.assert_called_once_with(
                    repo_id="openai/whisper-large-v3-turbo",
                    revision="41f01f3fe87f28c78e2fbf8b568835947dd65ed9",
                    cache_dir=str(
                        (Path(temp_dir) / "whisper-mps-turbo" / "transformers").resolve()
                    ),
                    ignore_patterns=[
                        "*.bin",
                        "*.h5",
                        "*.msgpack",
                        "*.onnx",
                        "*.tflite",
                        "model.fp32-*",
                    ],
                )

    def test_nemotron_downloader_places_pinned_gguf_in_model_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(os.environ, {"SONA_HF_CACHE": temp_dir}, clear=False):
                entry = model_download_service._entry("nemotron-3.5")

                def fake_download(*, repo_id, filename, revision, local_dir):
                    self.assertEqual(repo_id, "nvidia/nemotron-3.5-asr-streaming-0.6b")
                    self.assertEqual(filename, "nemotron-3.5-asr-streaming-0.6b.q8_0.gguf")
                    self.assertEqual(
                        revision,
                        "1c8deaecc64b91f034d73e08dd8b64625eb3395d",
                    )
                    target = Path(local_dir) / filename
                    target.write_bytes(b"gguf")
                    return str(target)

                with (
                    patch("huggingface_hub.hf_hub_download", fake_download),
                    patch(
                        "sona_ai.transcription.nemotron_transcriber.NEMOTRON_GGUF_SIZE_BYTES",
                        4,
                    ),
                ):
                    _download_nemotron(entry)

                self.assertTrue(
                    (
                        Path(temp_dir)
                        / "nemotron-3.5"
                        / "nemotron-3.5-asr-streaming-0.6b.q8_0.gguf"
                    ).is_file()
                )

    def test_uninstall_wav2vec2_removes_manifest_and_cache_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(os.environ, {"SONA_HF_CACHE": temp_dir}, clear=False):
                entry = model_download_service._entry("wav2vec2-aligner")
                cache_path = Path(temp_dir) / "wav2vec2-aligner"
                cache_path.mkdir(parents=True, exist_ok=True)
                (cache_path / "wav2vec2-align").mkdir(parents=True, exist_ok=True)
                (cache_path / "wav2vec2-align" / "marker.txt").write_text("ok")
                model_download_service.mark_installed("wav2vec2-aligner")

                job = self.wait_for_job(
                    model_download_service.start_uninstall("wav2vec2-aligner").job_id,
                )
                self.assertEqual(job.status, "uninstalled")
                self.assertFalse(cache_path.exists())
                self.assertFalse((model_manifest_dir() / "wav2vec2-aligner.json").exists())
                self.assertEqual(entry.runtime_cache_subdir, "wav2vec2-align")

    def test_redownload_faster_whisper_clears_only_selected_model_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(os.environ, {"SONA_HF_CACHE": temp_dir}, clear=False):
                selected_cache = Path(temp_dir) / "faster-whisper-large-v3"
                selected_cache.mkdir(parents=True, exist_ok=True)
                (selected_cache / "old.bin").write_text("old")
                sibling_cache = Path(temp_dir) / "faster-whisper-turbo"
                sibling_cache.mkdir(parents=True, exist_ok=True)
                (sibling_cache / "keep.bin").write_text("keep")
                model_download_service.mark_installed("faster-whisper-large-v3")
                model_download_service.mark_installed("faster-whisper-turbo")

                def fake_downloader(entry):
                    selected_cache.mkdir(parents=True, exist_ok=True)
                    (selected_cache / f"{entry.id}.bin").write_text("new")

                with patch.dict(
                    "sona_ai.services.model_download_service.DOWNLOADERS",
                    {"faster-whisper-large-v3": fake_downloader},
                    clear=False,
                ):
                    job = self.wait_for_job(
                        model_download_service.start_redownload("faster-whisper-large-v3").job_id,
                    )

                self.assertEqual(job.status, "installed")
                self.assertFalse((selected_cache / "old.bin").exists())
                self.assertTrue((selected_cache / "faster-whisper-large-v3.bin").exists())
                self.assertTrue((sibling_cache / "keep.bin").exists())
                self.assertTrue((model_manifest_dir() / "faster-whisper-large-v3.json").exists())
                self.assertTrue((model_manifest_dir() / "faster-whisper-turbo.json").exists())

    def test_uninstall_parakeet_removes_isolated_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(os.environ, {"SONA_HF_CACHE": temp_dir}, clear=False):
                cache_path = Path(temp_dir) / "parakeet"
                (cache_path / "nemo").mkdir(parents=True, exist_ok=True)
                (cache_path / "nemo" / "marker.txt").write_text("ok")
                model_download_service.mark_installed("parakeet")

                job = self.wait_for_job(
                    model_download_service.start_uninstall("parakeet").job_id,
                )

                self.assertEqual(job.status, "uninstalled")
                self.assertFalse(cache_path.exists())
                self.assertFalse((model_manifest_dir() / "parakeet.json").exists())

    def test_managed_model_root_uses_per_model_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(os.environ, {"SONA_HF_CACHE": temp_dir}, clear=False):
                self.assertEqual(
                    model_cache_root(model_id="parakeet").resolve(),
                    (Path(temp_dir) / "parakeet").resolve(),
                )
                self.assertEqual(
                    model_cache_root({"_sona_managed_model_id": "wav2vec2-aligner"}).resolve(),
                    (Path(temp_dir) / "wav2vec2-aligner").resolve(),
                )

    def test_model_environment_keeps_numba_cache_in_writable_model_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(os.environ, {"SONA_HF_CACHE": temp_dir}, clear=False):
                cache_dir = setup_model_cache_environment(model_id="parakeet")

                self.assertEqual(
                    Path(os.environ["NUMBA_CACHE_DIR"]).resolve(),
                    (cache_dir / "numba").resolve(),
                )
                self.assertTrue(cache_dir.is_dir())


if __name__ == "__main__":
    unittest.main()
