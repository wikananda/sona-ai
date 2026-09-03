import unittest

from sona_ai.transcription.nemotron_live_protocol import (
    NemotronLiveProtocolError,
    NemotronLiveTranscriptAccumulator,
)


class NemotronLiveProtocolTest(unittest.TestCase):
    def test_append_only_deltas_become_one_replaceable_provisional(self):
        accumulator = NemotronLiveTranscriptAccumulator("session", "en-US")
        accumulator.set_audio_end(1.0)

        first = accumulator.apply_delta({"item_id": "item-1", "delta": "Hello"})
        second = accumulator.apply_delta({"item_id": "item-1", "delta": " world"})

        self.assertEqual(first["provisional"]["text"], "Hello")
        self.assertEqual(second["provisional"]["text"], "Hello world")
        self.assertEqual(second["provisional"]["end"], 1.0)
        self.assertEqual(second["committed"], [])

    def test_completion_is_authoritative_and_normalizes_words(self):
        accumulator = NemotronLiveTranscriptAccumulator("session")
        accumulator.set_audio_end(1.5)
        accumulator.apply_delta({"item_id": "item-1", "delta": "helo"})

        event = accumulator.apply_completed({
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "item-1",
            "transcript": "Hello world.",
            "language": "en-US",
            "words": [
                {"word": "Hello", "start": 0.1, "end": 0.5, "confidence": 0.8},
                {"word": "world", "start": 0.6, "end": 1.1, "confidence": 2},
            ],
        })

        self.assertIsNone(event["provisional"])
        self.assertEqual(event["language"], "en-US")
        self.assertEqual(event["committed"][0]["text"], "Hello world.")
        self.assertEqual(event["committed"][0]["words"][1]["score"], 1.0)

        final = accumulator.finalize()
        self.assertEqual(final["type"], "final")
        self.assertEqual(final["segments"], event["committed"])

    def test_endpointed_relative_word_times_are_shifted_after_prior_item(self):
        accumulator = NemotronLiveTranscriptAccumulator("session", "en-US")
        accumulator.set_audio_end(1.0)
        accumulator.apply_completed({
            "item_id": "one",
            "transcript": "One.",
            "words": [{"word": "One", "start": 0.2, "end": 0.9}],
        })
        accumulator.set_audio_end(2.0)

        event = accumulator.apply_completed({
            "item_id": "two",
            "transcript": "Two.",
            "words_info": {
                "words": [{"word": "Two", "start_time": 0.1, "end_time": 0.8}],
            },
        })

        self.assertAlmostEqual(event["committed"][0]["start"], 1.0)
        self.assertAlmostEqual(event["committed"][0]["end"], 1.7)

    def test_duplicate_completion_is_ignored(self):
        accumulator = NemotronLiveTranscriptAccumulator("session", "en-US")
        completion = {"item_id": "same", "transcript": "Once."}

        self.assertIsNotNone(accumulator.apply_completed(completion))
        self.assertIsNone(accumulator.apply_completed(completion))
        self.assertEqual(len(accumulator.finalize()["segments"]), 1)

    def test_rejects_non_string_delta_and_item_switch(self):
        accumulator = NemotronLiveTranscriptAccumulator("session")
        with self.assertRaises(NemotronLiveProtocolError):
            accumulator.apply_delta({"delta": 123})

        accumulator.apply_delta({"item_id": "one", "delta": "hello"})
        with self.assertRaises(NemotronLiveProtocolError):
            accumulator.apply_delta({"item_id": "two", "delta": "world"})


if __name__ == "__main__":
    unittest.main()
