import tempfile
import unittest
import wave
from pathlib import Path

from sona_ai.transcription.live_timestamps import segment_live_timestamps
from sona_ai.transcription.schemas import (
    TranscriptSegment,
    TranscriptionResult,
    WordSegment,
)


class LiveTimestampingTest(unittest.TestCase):
    def test_splits_single_segment_by_sentence_and_audio_duration(self):
        audio_path = self._wav_file(duration_seconds=9.0)
        result = TranscriptionResult(
            segments=[
                TranscriptSegment(
                    text="First sentence. Second sentence? Third sentence!",
                    start=0.0,
                    end=0.0,
                )
            ],
            language="en",
        )

        segmented = segment_live_timestamps(result, audio_path)

        self.assertEqual(len(segmented.segments), 3)
        self.assertEqual(segmented.segments[0].text, "First sentence.")
        self.assertEqual(segmented.segments[-1].text, "Third sentence!")
        self.assertEqual(segmented.segments[0].start, 0.0)
        self.assertAlmostEqual(segmented.segments[-1].end, 9.0)
        self.assertLess(segmented.segments[0].end, segmented.segments[1].end)

    def test_groups_timed_words_by_sentence_boundary(self):
        result = TranscriptionResult(
            segments=[
                TranscriptSegment(
                    text="Hello world. Next line.",
                    start=0.0,
                    end=2.0,
                    words=[
                        WordSegment("Hello", 0.0, 0.3),
                        WordSegment("world.", 0.3, 0.8),
                        WordSegment("Next", 1.0, 1.3),
                        WordSegment("line.", 1.3, 1.8),
                    ],
                )
            ],
            language="en",
        )

        segmented = segment_live_timestamps(result, "missing.wav")

        self.assertEqual(len(segmented.segments), 2)
        self.assertEqual(segmented.segments[0].text, "Hello world.")
        self.assertEqual(segmented.segments[0].start, 0.0)
        self.assertEqual(segmented.segments[0].end, 0.8)
        self.assertEqual(segmented.segments[1].text, "Next line.")
        self.assertEqual(len(segmented.segments[1].words), 2)

    def test_keeps_existing_multiple_timed_segments(self):
        result = TranscriptionResult(
            segments=[
                TranscriptSegment("One.", 0.0, 1.0),
                TranscriptSegment("Two.", 1.0, 2.0),
            ],
            language="en",
        )

        segmented = segment_live_timestamps(result, "missing.wav")

        self.assertIs(segmented, result)

    def _wav_file(self, duration_seconds: float) -> str:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        path = Path(temp_dir.name) / "audio.wav"
        sample_rate = 16000
        frames = int(duration_seconds * sample_rate)
        with wave.open(str(path), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(sample_rate)
            audio.writeframes(b"\x00\x00" * frames)
        return str(path)


if __name__ == "__main__":
    unittest.main()
