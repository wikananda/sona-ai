import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sona_ai.core import model_manifest_dir
from sona_ai.services.model_download_service import model_download_service
from sona_ai.services.pipeline_profile import PipelineProfile


class ModelDownloadServiceTest(unittest.TestCase):
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
                    Path(temp_dir).resolve(),
                )


if __name__ == "__main__":
    unittest.main()
