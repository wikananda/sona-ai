import unittest

from sona_ai.transcription.nemotron_languages import (
    NEMOTRON_LOCALES,
    resolve_nemotron_language,
)


class NemotronLanguagesTest(unittest.TestCase):
    def test_auto_and_bare_language_codes_are_normalized(self):
        self.assertEqual(resolve_nemotron_language(None), "auto")
        self.assertEqual(resolve_nemotron_language("AUTO"), "auto")
        self.assertEqual(resolve_nemotron_language("en"), "en-US")
        self.assertEqual(resolve_nemotron_language("pt"), "pt-BR")
        self.assertEqual(resolve_nemotron_language("ZH_cn"), "zh-CN")

    def test_all_published_out_of_box_locales_are_accepted(self):
        self.assertEqual(len(NEMOTRON_LOCALES), 32)
        for locale in NEMOTRON_LOCALES:
            with self.subTest(locale=locale):
                self.assertEqual(resolve_nemotron_language(locale.lower()), locale)

    def test_indonesian_and_adaptation_only_locales_are_rejected(self):
        for language in ("id", "id-ID", "th-TH", "he-IL"):
            with self.subTest(language=language):
                with self.assertRaisesRegex(ValueError, "does not support"):
                    resolve_nemotron_language(language)


if __name__ == "__main__":
    unittest.main()
