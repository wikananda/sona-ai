import unittest
from unittest.mock import patch

import numpy as np

from sona_ai.services.transcription_service import TranscriptionService
from sona_ai.transcription.schemas import TranscriptionResult


SPEECH_CONFIG = {
    "transcription": {
        "engine": "parakeet",
        "config": "parakeet",
        "device": "cpu",
    },
    "alignment": {
        "enabled": True,
        "engine": "wav2vec2_external",
        "config": "wav2vec2",
    },
    "diarization": {
        "enabled": True,
        "engine": "community_external",
        "config": "diarization-community",
    },
}


class FakeTranscriber:
    device = "cpu"

    def __init__(self):
        self.calls = []

    def transcribe_samples(self, samples, language=None):
        self.calls.append((samples, language))
        return TranscriptionResult(segments=[], language=language)


class FakePipeline:
    def __init__(self, transcriber):
        self.transcriber = transcriber
        self.cleaned = False

    def cleanup_models(self):
        self.cleaned = True


class TranscriptionServiceLiveTest(unittest.TestCase):
    def make_service(self):
        transcriber = FakeTranscriber()
        pipeline = FakePipeline(transcriber)
        with patch(
            "sona_ai.services.transcription_service.model_download_service.mark_profile_installed"
        ):
            service = TranscriptionService(
                pipeline,
                speech_config=SPEECH_CONFIG,
                default_model="parakeet",
                default_device="cpu",
            )
        return service, pipeline, transcriber

    def test_live_profile_reuses_loaded_transcriber_despite_pipeline_options(self):
        service, _, transcriber = self.make_service()
        samples = np.zeros(320, dtype=np.float32)

        with patch.object(
            service,
            "_get_pipeline",
            side_effect=AssertionError("must not load a duplicate model"),
        ):
            service.prepare_live_transcription(model="parakeet", device="cpu")
            result = service.transcribe_live_samples(
                samples,
                language="en",
                model="parakeet",
                device="cpu",
            )

        self.assertEqual(result.language, "en")
        self.assertIs(transcriber.calls[0][0], samples)

    def test_close_cleans_pipeline(self):
        service, pipeline, _ = self.make_service()

        service.close()

        self.assertTrue(pipeline.cleaned)


if __name__ == "__main__":
    unittest.main()
