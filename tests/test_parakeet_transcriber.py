from types import SimpleNamespace
import unittest

import numpy as np

from sona_ai.transcription.parakeet_transcriber import ParakeetTranscriber


def transcriber_config():
    return {
        "model": {
            "model_name": "nvidia/parakeet-tdt-0.6b-v3",
            "language": "en",
            "supported_languages": ["en"],
            "device": "cpu",
            "batch_size": 1,
        },
    }


class FakeParakeetModel:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio, **kwargs):
        self.calls.append((audio, kwargs))
        hypothesis = SimpleNamespace(
            text="Hello world.",
            timestep={
                "word": [
                    {"word": "Hello", "start": 0.1, "end": 0.4},
                    {"word": "world.", "start": 0.4, "end": 0.8},
                ],
                "segment": [{
                    "segment": "Hello world.",
                    "start": 0.1,
                    "end": 0.8,
                }],
            },
        )
        return [[hypothesis]]


class ParakeetTranscriberTest(unittest.TestCase):
    def test_transcribes_float32_samples_without_file_conversion(self):
        transcriber = ParakeetTranscriber(transcriber_config())
        transcriber.model = FakeParakeetModel()
        samples = np.zeros(16000, dtype=np.float32)

        result = transcriber.transcribe_samples(samples, language="en")

        audio, kwargs = transcriber.model.calls[0]
        self.assertIs(audio, samples)
        self.assertTrue(kwargs["timestamps"])
        self.assertFalse(kwargs["verbose"])
        self.assertEqual(result.segments[0].text, "Hello world.")
        self.assertEqual(result.segments[0].words[1].word, "world.")
        self.assertEqual(result.segments[0].words[1].end, 0.8)

    def test_rejects_non_mono_or_non_finite_samples(self):
        transcriber = ParakeetTranscriber(transcriber_config())
        transcriber.model = FakeParakeetModel()

        with self.assertRaises(ValueError):
            transcriber.transcribe_samples(np.zeros((2, 4), dtype=np.float32))
        with self.assertRaises(ValueError):
            transcriber.transcribe_samples(np.asarray([np.nan], dtype=np.float32))

    def test_empty_samples_return_empty_result_without_model_call(self):
        transcriber = ParakeetTranscriber(transcriber_config())
        transcriber.model = FakeParakeetModel()

        result = transcriber.transcribe_samples(np.asarray([], dtype=np.float32))

        self.assertEqual(result.segments, [])
        self.assertEqual(transcriber.model.calls, [])


if __name__ == "__main__":
    unittest.main()
