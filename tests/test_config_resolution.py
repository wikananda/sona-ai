import unittest
from copy import deepcopy

from sona_ai.core import load_config
from sona_ai.pipelines import build_speech_pipeline
from sona_ai.services.summarization_service import SummarizationService


class ConfigResolutionTest(unittest.TestCase):
    def test_speech_devices_default_from_speech_config(self):
        speech_config = deepcopy(load_config("speech"))
        speech_config["transcription"] = {
            "engine": "parakeet",
            "config": "parakeet",
            "device": "cpu",
        }
        speech_config["alignment"]["device"] = "cpu"
        speech_config["diarization"]["device"] = "cpu"

        pipeline = build_speech_pipeline(speech_config, write_outputs=False)

        self.assertEqual(pipeline.transcriber.device, "cpu")
        self.assertEqual(pipeline.aligner.device, "cpu")
        self.assertEqual(pipeline.diarizer.device, "cpu")

    def test_request_device_overrides_transcription_and_alignment(self):
        speech_config = deepcopy(load_config("speech"))
        speech_config["transcription"] = {
            "engine": "faster_whisper",
            "config": "faster-whisper-turbo",
            "device": "cpu",
        }
        speech_config["alignment"]["device"] = "cpu"

        pipeline = build_speech_pipeline(
            speech_config,
            device="auto",
            write_outputs=False,
        )

        self.assertIn(pipeline.transcriber.device, {"cpu", "cuda"})
        self.assertIn(pipeline.aligner.device, {"cpu", "mps", "cuda"})

    def test_diarization_uses_speech_config_device(self):
        speech_config = deepcopy(load_config("speech"))
        speech_config["diarization"]["device"] = "cpu"

        pipeline = build_speech_pipeline(
            speech_config,
            device="auto",
            write_outputs=False,
        )

        self.assertEqual(pipeline.diarizer.device, "cpu")

    def test_summarization_limits_default_from_selected_model_config(self):
        service = SummarizationService()

        self.assertEqual(service._model_input_limit("qwen"), 2048)
        self.assertEqual(service._model_input_limit("llama"), 512)
        self.assertEqual(service._model_output_limit(load_config("gemma")), 256)

    def test_summarization_constructor_output_override_wins(self):
        service = SummarizationService(max_new_tokens=99)

        self.assertEqual(service._model_output_limit(load_config("qwen")), 99)


if __name__ == "__main__":
    unittest.main()
