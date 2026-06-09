from types import SimpleNamespace
import unittest

from sona_ai.services.pipeline_profile import resolve_pipeline_profile
from sona_ai.transcription.faster_whisper_transcriber import FasterWhisperTranscriber


def transcriber_config(device: str = "cpu") -> dict:
    return {
        "model": {
            "model_name": "turbo",
            "language": None,
            "device": device,
            "compute_type": None,
            "cpu_threads": 0,
            "num_workers": 1,
            "task": "transcribe",
            "beam_size": 5,
            "vad_filter": False,
            "word_timestamps": False,
        },
        "cp_dir": {"hf_cache": ".models"},
    }


class FakeFasterWhisperModel:
    def transcribe(self, audio_path, **kwargs):
        words = [
            SimpleNamespace(word="Hello", start=0.0, end=0.4, probability=0.9),
            SimpleNamespace(word="world", start=0.4, end=0.8, probability=0.8),
        ]
        segments = [
            SimpleNamespace(start=0.0, end=0.8, text=" Hello world ", words=words),
        ]
        info = SimpleNamespace(language="en", language_probability=0.99, duration=1.0)
        return iter(segments), info


class FasterWhisperTranscriberTest(unittest.TestCase):
    def test_transcribe_converts_segments(self):
        transcriber = FasterWhisperTranscriber(transcriber_config())
        transcriber.model = FakeFasterWhisperModel()

        result = transcriber.transcribe("audio.mp3")

        self.assertEqual(result.language, "en")
        self.assertEqual(len(result.segments), 1)
        self.assertEqual(result.segments[0].text, "Hello world")
        self.assertEqual(result.segments[0].start, 0.0)
        self.assertEqual(result.segments[0].end, 0.8)
        self.assertEqual(result.segments[0].words[0].score, 0.9)

    def test_mps_falls_back_to_cpu(self):
        transcriber = FasterWhisperTranscriber(transcriber_config(device="mps"))

        self.assertEqual(transcriber.device, "cpu")
        self.assertEqual(transcriber.compute_type, "int8")

    def test_model_profile_routes_to_faster_whisper_config(self):
        speech_config = {
            "transcription": {"engine": "parakeet", "config": "parakeet"},
            "alignment": {"enabled": True, "engine": "wav2vec2_external", "config": "wav2vec2"},
            "diarization": {
                "enabled": True,
                "engine": "community_external",
                "config": "diarization-community",
            },
        }

        profile = resolve_pipeline_profile(
            speech_config,
            model="faster-whisper-large-v3",
            device="cpu",
        )

        self.assertEqual(profile.transcription_engine, "faster_whisper")
        self.assertEqual(profile.transcription_config, "faster-whisper-large-v3")

    def test_live_profile_can_disable_alignment_and_diarization(self):
        speech_config = {
            "transcription": {"engine": "parakeet", "config": "parakeet"},
            "alignment": {"enabled": True, "engine": "wav2vec2_external", "config": "wav2vec2"},
            "diarization": {
                "enabled": True,
                "engine": "community_external",
                "config": "diarization-community",
            },
        }

        profile = resolve_pipeline_profile(
            speech_config,
            model="parakeet",
            device="cpu",
            alignment_enabled=False,
            diarization_enabled=False,
        )

        self.assertFalse(profile.alignment_enabled)
        self.assertFalse(profile.diarization_enabled)


if __name__ == "__main__":
    unittest.main()
