import unittest

from sona_ai.transcription.parakeet_live_protocol import (
    ParakeetLiveProtocolError,
    ParakeetLiveTranscriptAccumulator,
)


def segment(*words):
    return [{
        "text": " ".join(word[0] for word in words),
        "start": words[0][1] if words else 0,
        "end": words[-1][2] if words else 0,
        "words": [
            {"word": text, "start": start, "end": end}
            for text, start, end in words
        ],
    }]


class ParakeetLiveTranscriptAccumulatorTest(unittest.TestCase):
    def setUp(self):
        self.accumulator = ParakeetLiveTranscriptAccumulator("session-1", "en")

    def test_commits_stable_words_and_replaces_right_context(self):
        first = self.accumulator.apply_snapshot(
            segment(
                ("And", 0.2, 0.6),
                ("so,", 0.6, 0.9),
                ("my", 1.0, 1.3),
                ("fellow", 1.3, 1.8),
                ("Americans,", 1.8, 2.7),
            ),
            window_start=0,
            stable_cutoff=2,
            audio_end=4,
        )
        second = self.accumulator.apply_snapshot(
            segment(
                ("And", 0.22, 0.62),
                ("so,", 0.62, 0.92),
                ("my", 1.02, 1.32),
                ("fellow", 1.32, 1.82),
                ("Americans,", 1.82, 2.72),
                ("ask", 2.8, 3.2),
                ("not", 3.2, 3.6),
                ("what", 3.8, 4.2),
            ),
            window_start=0,
            stable_cutoff=4,
            audio_end=6,
        )
        third = self.accumulator.apply_snapshot(
            segment(
                ("And", 0.21, 0.61),
                ("so,", 0.61, 0.91),
                ("my", 1.01, 1.31),
                ("fellow", 1.31, 1.81),
                ("Americans,", 1.81, 2.71),
                ("ask", 2.81, 3.21),
                ("not", 3.21, 3.61),
                ("what", 3.81, 4.21),
            ),
            window_start=0,
            stable_cutoff=6,
            audio_end=8,
        )

        self.assertEqual(first["committed"], [])
        self.assertEqual(first["provisional"]["text"], "And so, my fellow Americans,")
        self.assertEqual(second["committed"][0]["text"], "And so, my fellow")
        self.assertEqual(second["provisional"]["text"], "Americans, ask not what")
        self.assertEqual(third["committed"][0]["text"], "Americans, ask not")
        self.assertEqual(third["provisional"]["text"], "what")
        self.assertEqual(
            [word["word"] for item in self.accumulator.committed for word in item["words"]],
            ["And", "so,", "my", "fellow", "Americans,", "ask", "not"],
        )

    def test_window_offsets_become_absolute_timestamps(self):
        first = self.accumulator.apply_snapshot(
            segment(("later", 9.0, 9.5)),
            window_start=10,
            stable_cutoff=20,
            audio_end=22,
        )
        event = self.accumulator.apply_snapshot(
            segment(("later", 9.05, 9.55)),
            window_start=10,
            stable_cutoff=21,
            audio_end=23,
        )

        self.assertEqual(first["committed"], [])
        word = event["committed"][0]["words"][0]
        self.assertEqual(word["start"], 19.05)
        self.assertEqual(word["end"], 19.55)

    def test_silence_advances_horizon_and_prevents_old_word_reappearing(self):
        first = self.accumulator.apply_snapshot(
            segment(("hello", 0.2, 0.8)),
            window_start=0,
            stable_cutoff=2,
            audio_end=4,
        )
        silence = self.accumulator.apply_snapshot(
            [],
            window_start=0,
            stable_cutoff=4,
            audio_end=6,
        )
        repeated = self.accumulator.apply_snapshot(
            segment(("hello", 0.25, 0.85)),
            window_start=0,
            stable_cutoff=6,
            audio_end=8,
        )

        self.assertEqual(first["committed"], [])
        self.assertEqual(first["provisional"]["text"], "hello")
        self.assertIsNotNone(silence)
        self.assertIsNone(silence["provisional"])
        self.assertIsNone(repeated)
        self.assertEqual(self.accumulator.commit_horizon, 6)

    def test_word_crossing_cutoff_commits_on_next_snapshot(self):
        first = self.accumulator.apply_snapshot(
            segment(("boundary", 1.8, 2.2)),
            window_start=0,
            stable_cutoff=2,
            audio_end=4,
        )
        second = self.accumulator.apply_snapshot(
            segment(("boundary", 1.82, 2.22)),
            window_start=0,
            stable_cutoff=4,
            audio_end=6,
        )
        third = self.accumulator.apply_snapshot(
            segment(("boundary", 1.81, 2.21)),
            window_start=0,
            stable_cutoff=6,
            audio_end=8,
        )

        self.assertEqual(first["committed"], [])
        self.assertEqual(first["provisional"]["text"], "boundary")
        self.assertEqual(second["committed"], [])
        self.assertEqual(second["provisional"]["text"], "boundary")
        self.assertEqual(third["committed"][0]["text"], "boundary")
        self.assertIsNone(third["provisional"])

    def test_revised_stable_word_waits_for_repeat_before_commit(self):
        self.accumulator.apply_snapshot(
            segment(("your", 1.4, 1.8), ("relation", 2.2, 2.8)),
            window_start=0,
            stable_cutoff=2,
            audio_end=4,
        )
        second = self.accumulator.apply_snapshot(
            segment(("your", 1.4, 1.8), ("relation", 2.2, 2.8)),
            window_start=0,
            stable_cutoff=4,
            audio_end=6,
        )
        third = self.accumulator.apply_snapshot(
            segment(("your", 1.4, 1.8), ("relationship", 2.2, 3.0)),
            window_start=0,
            stable_cutoff=6,
            audio_end=8,
        )
        fourth = self.accumulator.apply_snapshot(
            segment(("your", 1.4, 1.8), ("relationship", 2.2, 3.0)),
            window_start=0,
            stable_cutoff=8,
            audio_end=10,
        )

        self.assertEqual(second["committed"][0]["text"], "your")
        self.assertEqual(second["provisional"]["text"], "relation")
        self.assertEqual(third["committed"], [])
        self.assertEqual(third["provisional"]["text"], "relationship")
        self.assertEqual(fourth["committed"][0]["text"], "relationship")

    def test_missing_timestamps_never_become_committed(self):
        event = self.accumulator.apply_snapshot(
            [{"text": "unsafe", "words": [{"word": "unsafe"}]}],
            window_start=0,
            stable_cutoff=2,
            audio_end=4,
        )

        self.assertIsNone(event)
        self.assertEqual(self.accumulator.committed, [])

    def test_final_promotes_remaining_provisional_once(self):
        self.accumulator.apply_snapshot(
            segment(("last", 2.1, 2.5)),
            window_start=0,
            stable_cutoff=2,
            audio_end=3,
        )

        final = self.accumulator.finalize()

        self.assertEqual(final["type"], "final")
        self.assertEqual([item["text"] for item in final["segments"]], ["last"])
        self.assertEqual([item["text"] for item in final["committed"]], ["last"])
        self.assertIsNone(final["provisional"])

    def test_final_snapshot_commits_without_waiting_for_confirmation(self):
        event = self.accumulator.apply_snapshot(
            segment(("last", 0.2, 0.6)),
            window_start=0,
            stable_cutoff=1,
            audio_end=1,
            final=True,
        )

        self.assertEqual(event["committed"][0]["text"], "last")
        self.assertIsNone(event["provisional"])

    def test_rejects_invalid_snapshot_bounds(self):
        with self.assertRaises(ParakeetLiveProtocolError):
            self.accumulator.apply_snapshot(
                [],
                window_start=0,
                stable_cutoff=3,
                audio_end=2,
            )


if __name__ == "__main__":
    unittest.main()
