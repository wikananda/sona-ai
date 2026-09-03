import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from sona_ai.transcription.whisper_mps_transcriber import WhisperMpsTranscriber


def transcriber_config() -> dict:
    return {
        "model": {
            "model_name": "openai/whisper-large-v3-turbo",
            "revision": "revision-1",
            "device": "mps",
            "dtype": "float16",
            "task": "transcribe",
            "batch_size": 4,
            "live_batch_size": 1,
            "chunk_length_s": 30,
            "word_timestamps": True,
            "warmup_seconds": 0,
        },
        "cp_dir": {"hf_cache": ".models"},
    }


class FakePipeline:
    device = SimpleNamespace(type="mps")

    def __init__(self):
        self.calls = []

    def __call__(self, audio, **kwargs):
        self.calls.append((audio, kwargs))
        return {
            "text": " Hello, world. Another sentence",
            "chunks": [
                {"text": " Hello", "timestamp": (0.0, 0.4)},
                {"text": ",", "timestamp": (0.4, 0.5)},
                {"text": " world.", "timestamp": (0.5, 0.9)},
                {"text": " Another", "timestamp": (2.1, 2.5)},
                {"text": " sentence", "timestamp": (2.5, 3.0)},
            ],
        }


class OomThenSuccessPipeline(FakePipeline):
    def __init__(self):
        super().__init__()
        self.batch_sizes = []

    def __call__(self, audio, **kwargs):
        self.batch_sizes.append(kwargs["batch_size"])
        if len(self.batch_sizes) == 1:
            raise RuntimeError("MPS backend out of memory")
        return super().__call__(audio, **kwargs)


class WhisperMpsTranscriberTest(unittest.TestCase):
    def test_load_pins_revision_and_verifies_pipeline_device(self):
        transcriber = WhisperMpsTranscriber(transcriber_config())
        model = MagicMock()
        processor = SimpleNamespace(tokenizer="tokenizer", feature_extractor="features")
        inference_pipeline = FakePipeline()

        with (
            patch("torch.backends.mps.is_available", return_value=True),
            patch("transformers.AutoModelForSpeechSeq2Seq.from_pretrained", return_value=model) as load_model,
            patch("transformers.AutoProcessor.from_pretrained", return_value=processor) as load_processor,
            patch("transformers.pipeline", return_value=inference_pipeline) as create_pipeline,
            patch.object(transcriber, "_warm_up"),
        ):
            transcriber.load_models()

        self.assertIs(transcriber.pipeline, inference_pipeline)
        self.assertEqual(load_model.call_args.kwargs["revision"], "revision-1")
        self.assertEqual(load_processor.call_args.kwargs["revision"], "revision-1")
        model.to.assert_called_once()
        self.assertEqual(model.to.call_args.args[0].type, "mps")
        self.assertEqual(create_pipeline.call_args.kwargs["device"].type, "mps")

    def test_whole_file_uses_batched_word_timestamp_inference(self):
        transcriber = WhisperMpsTranscriber(transcriber_config())
        transcriber.pipeline = FakePipeline()

        result = transcriber.transcribe("meeting.wav", language="id")

        _, kwargs = transcriber.pipeline.calls[0]
        self.assertEqual(kwargs["batch_size"], 4)
        self.assertEqual(kwargs["return_timestamps"], "word")
        self.assertEqual(kwargs["generate_kwargs"], {"task": "transcribe", "language": "id"})
        self.assertEqual(result.language, "id")
        self.assertEqual([segment.text for segment in result.segments], [
            "Hello, world.",
            "Another sentence",
        ])
        self.assertEqual(result.segments[0].words[0].word, "Hello")

    def test_live_samples_use_single_batch_and_16khz_input(self):
        transcriber = WhisperMpsTranscriber(transcriber_config())
        transcriber.pipeline = FakePipeline()
        samples = np.full(1600, 0.1, dtype=np.float64)

        transcriber.transcribe_samples(samples)

        audio, kwargs = transcriber.pipeline.calls[0]
        self.assertEqual(kwargs["batch_size"], 1)
        self.assertEqual(audio["sampling_rate"], 16000)
        self.assertEqual(audio["array"].dtype, np.float32)
        self.assertTrue(audio["array"].flags.c_contiguous)

    def test_live_silence_does_not_invoke_whisper(self):
        transcriber = WhisperMpsTranscriber(transcriber_config())
        transcriber.pipeline = FakePipeline()

        result = transcriber.transcribe_samples(np.zeros(1600, dtype=np.float32), language="en")

        self.assertEqual(result.segments, [])
        self.assertEqual(result.language, "en")
        self.assertEqual(transcriber.pipeline.calls, [])

    def test_whole_file_retries_mps_out_of_memory_with_one_batch(self):
        transcriber = WhisperMpsTranscriber(transcriber_config())
        pipeline = OomThenSuccessPipeline()
        transcriber.pipeline = pipeline

        with patch("torch.mps.empty_cache"):
            transcriber.transcribe("meeting.wav")

        self.assertEqual(pipeline.batch_sizes, [4, 1])

    def test_live_samples_reject_non_finite_audio(self):
        transcriber = WhisperMpsTranscriber(transcriber_config())
        transcriber.pipeline = FakePipeline()

        with self.assertRaisesRegex(ValueError, "non-finite"):
            transcriber.transcribe_samples(np.array([np.nan], dtype=np.float32))

    def test_load_fails_instead_of_silently_using_cpu(self):
        transcriber = WhisperMpsTranscriber(transcriber_config())

        with patch("torch.backends.mps.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "MPS is unavailable"):
                transcriber.load_models()


if __name__ == "__main__":
    unittest.main()
