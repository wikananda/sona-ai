import unittest

from sona_ai.alignment.external_wav2vec2_aligner import ExternalWav2Vec2Aligner
from sona_ai.transcription.schemas import TranscriptSegment, TranscriptionResult


class ExternalWav2Vec2AlignerTest(unittest.TestCase):
    def test_keeps_original_when_alignment_has_no_timestamps(self):
        aligner = ExternalWav2Vec2Aligner(
            {
                "model": {
                    "align_model": "facebook/wav2vec2-base-960h",
                    "device": "cpu",
                },
                "alignment": {},
                "cp_dir": {"hf_cache": "cp/hf_cache"},
            }
        )
        original = TranscriptionResult(
            segments=[
                TranscriptSegment(text="Hello world.", start=1.0, end=2.0),
            ],
            language="en",
        )

        self.assertTrue(
            aligner._should_keep_original_transcription(
                original=original,
                aligned_timed_segments=0,
                aligned_timed_words=0,
            )
        )

    def test_uses_alignment_when_alignment_has_timestamps(self):
        aligner = ExternalWav2Vec2Aligner(
            {
                "model": {
                    "align_model": "facebook/wav2vec2-base-960h",
                    "device": "cpu",
                },
                "alignment": {},
                "cp_dir": {"hf_cache": "cp/hf_cache"},
            }
        )
        original = TranscriptionResult(
            segments=[
                TranscriptSegment(text="Hello world.", start=1.0, end=2.0),
            ],
            language="en",
        )

        self.assertFalse(
            aligner._should_keep_original_transcription(
                original=original,
                aligned_timed_segments=1,
                aligned_timed_words=0,
            )
        )


if __name__ == "__main__":
    unittest.main()
