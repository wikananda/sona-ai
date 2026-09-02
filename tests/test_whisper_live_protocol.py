import math
import unittest

from sona_ai.transcription.whisper_live_protocol import (
    WhisperLiveProtocolError,
    WhisperLiveTranscriptAccumulator,
)


class WhisperLiveTranscriptAccumulatorTest(unittest.TestCase):
    def setUp(self):
        self.accumulator = WhisperLiveTranscriptAccumulator("session-1")

    def test_deduplicates_repeated_committed_segments_and_replaces_partial(self):
        first = self.accumulator.apply_segments([
            {"start": "0.000", "end": "1.250", "text": "Hello", "completed": True},
            {"start": "1.250", "end": "2.000", "text": "wor", "completed": False},
        ])
        second = self.accumulator.apply_segments([
            {"start": "0.000", "end": "1.250", "text": "Hello", "completed": True},
            {"start": "1.250", "end": "2.400", "text": "world", "completed": False},
        ])

        self.assertEqual([segment["text"] for segment in first["committed"]], ["Hello"])
        self.assertEqual(first["provisional"]["text"], "wor")
        self.assertEqual(second["committed"], [])
        self.assertEqual(second["provisional"]["text"], "world")
        self.assertEqual([segment["text"] for segment in self.accumulator.committed], ["Hello"])

    def test_promotes_last_partial_when_stream_ends(self):
        self.accumulator.apply_segments([
            {"start": "0", "end": "1", "text": "Last words", "completed": False},
        ])

        final = self.accumulator.finalize()

        self.assertEqual(final["type"], "final")
        self.assertIsNone(final["provisional"])
        self.assertEqual([segment["text"] for segment in final["segments"]], ["Last words"])
        self.assertEqual([segment["text"] for segment in final["committed"]], ["Last words"])

    def test_normalizes_word_times_and_probability(self):
        event = self.accumulator.apply_segments([
            {
                "start": "2.0",
                "end": "3.0",
                "text": " hello ",
                "completed": True,
                "words": [{
                    "word": " hello",
                    "start": "1.0",
                    "end": "4.0",
                    "probability": "1.5",
                }],
            }
        ])

        word = event["committed"][0]["words"][0]
        self.assertEqual(word["start"], 2.0)
        self.assertEqual(word["end"], 3.0)
        self.assertEqual(word["score"], 1.0)

    def test_clamps_invalid_and_non_monotonic_times(self):
        event = self.accumulator.apply_segments([
            {"start": "-4", "end": "nan", "text": "Safe", "completed": True},
            {"start": math.inf, "end": "-2", "text": "Also safe", "completed": True},
        ])

        self.assertEqual(event["committed"][0]["start"], 0.0)
        self.assertEqual(event["committed"][0]["end"], 0.0)
        self.assertEqual(event["committed"][1]["start"], 0.0)
        self.assertEqual(event["committed"][1]["end"], 0.0)

    def test_language_changes_increment_revision_only_when_changed(self):
        first = self.accumulator.set_language("en")
        duplicate = self.accumulator.set_language("en")

        self.assertEqual(first["language"], "en")
        self.assertIsNone(duplicate)
        self.assertEqual(self.accumulator.revision, 1)

    def test_rejects_non_list_segments(self):
        with self.assertRaises(WhisperLiveProtocolError):
            self.accumulator.apply_segments({"text": "invalid"})


if __name__ == "__main__":
    unittest.main()
