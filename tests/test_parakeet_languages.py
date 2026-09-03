import unittest

from sona_ai.transcription.parakeet_languages import (
    PARAKEET_LANGUAGES,
    resolve_parakeet_language,
)


class ParakeetLanguagesTest(unittest.TestCase):
    def test_published_checkpoint_has_all_25_languages(self):
        self.assertEqual(len(PARAKEET_LANGUAGES), 25)
        self.assertIn("en", PARAKEET_LANGUAGES)
        self.assertIn("el", PARAKEET_LANGUAGES)
        self.assertIn("uk", PARAKEET_LANGUAGES)

    def test_auto_detection_and_locale_normalization(self):
        self.assertIsNone(resolve_parakeet_language(None))
        self.assertIsNone(resolve_parakeet_language("AUTO"))
        self.assertEqual(resolve_parakeet_language("PT_br"), "pt")

    def test_indonesian_and_non_european_languages_are_rejected(self):
        for language in ("id", "ja", "zh"):
            with self.subTest(language=language):
                with self.assertRaisesRegex(ValueError, "does not support"):
                    resolve_parakeet_language(language)


if __name__ == "__main__":
    unittest.main()
