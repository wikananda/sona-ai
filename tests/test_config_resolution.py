import unittest
from copy import deepcopy

from sona_ai.core import load_config
from sona_ai.pipelines import build_speech_pipeline
from sona_ai.services.summarization_service import SummarizationService
from sona_ai.summarization.prompts import parse_adaptive_summary_response


class ConfigResolutionTest(unittest.TestCase):
    def test_speech_devices_default_from_speech_config(self):
        speech_config = deepcopy(load_config("speech"))
        speech_config["transcription"] = {
            "engine": "parakeet",
            "config": "parakeet",
            "device": "cpu",
        }
        speech_config["alignment"]["device"] = "cpu"
        speech_config["diarization"]["device"] = "cpu"

        pipeline = build_speech_pipeline(speech_config, write_outputs=False)

        self.assertEqual(pipeline.transcriber.device, "cpu")
        self.assertEqual(pipeline.aligner.device, "cpu")
        self.assertEqual(pipeline.diarizer.device, "cpu")

    def test_request_device_overrides_transcription_and_alignment(self):
        speech_config = deepcopy(load_config("speech"))
        speech_config["transcription"] = {
            "engine": "faster_whisper",
            "config": "faster-whisper-turbo",
            "device": "cpu",
        }
        speech_config["alignment"]["device"] = "cpu"

        pipeline = build_speech_pipeline(
            speech_config,
            device="auto",
            write_outputs=False,
        )

        self.assertIn(pipeline.transcriber.device, {"cpu", "cuda"})
        self.assertIn(pipeline.aligner.device, {"cpu", "mps", "cuda"})

    def test_diarization_uses_speech_config_device(self):
        speech_config = deepcopy(load_config("speech"))
        speech_config["diarization"]["device"] = "cpu"

        pipeline = build_speech_pipeline(
            speech_config,
            device="auto",
            write_outputs=False,
        )

        self.assertEqual(pipeline.diarizer.device, "cpu")

    def test_summarization_limits_default_from_selected_model_config(self):
        service = SummarizationService()

        self.assertEqual(service._model_input_limit("qwen"), 2048)
        self.assertEqual(service._model_input_limit("llama"), 512)
        self.assertEqual(service._model_output_limit(load_config("gemma")), 1024)

    def test_summarization_constructor_output_override_wins(self):
        service = SummarizationService(max_new_tokens=99)

        self.assertEqual(service._model_output_limit(load_config("qwen")), 99)

    def test_adaptive_summary_parser_accepts_json(self):
        result = parse_adaptive_summary_response(
            """
            {
              "plan": {
                "recording_type": "interview",
                "format_name": "Interview Summary",
                "audience": "research team",
                "sections": ["Context", "Findings"],
                "style": "concise",
                "rationale": "Interview transcript."
              },
              "summary_markdown": "## Context\\nUseful summary."
            }
            """
        )

        self.assertEqual(result["format_name"], "Interview Summary")
        self.assertEqual(result["plan"]["sections"], ["Context", "Findings"])
        self.assertEqual(result["summary"], "## Context\n\nUseful summary.")

    def test_adaptive_summary_parser_accepts_tagged_envelope(self):
        result = parse_adaptive_summary_response(
            """
            <plan_json>
            {
              "recording_type": "meeting",
              "format_name": "Meeting Brief",
              "sections": ["Decisions", "Next Steps"]
            }
            </plan_json>

            <summary_markdown>
            ## Decisions
            The team agreed to ship.

            Next Steps
            Follow up with QA.
            </summary_markdown>
            """
        )

        self.assertEqual(result["format_name"], "Meeting Brief")
        self.assertEqual(
            result["summary"],
            "## Decisions\n"
            "\nThe team agreed to ship.\n\n"
            "## Next Steps\n"
            "\n"
            "Follow up with QA.",
        )

    def test_adaptive_summary_parser_accepts_partial_tagged_envelope(self):
        result = parse_adaptive_summary_response(
            """
            <plan_json>
            { "recording_type": "interview | general conversation", "format_name": "Biography Summary", "audience": "General Reader", "sections": ["Early Life and Music Background", "Education and Career"], "style": "Informal and conversational", "rationale": "The format choice is based on the conversational tone of the transcript, which lends itself well to a biography-style summary." }
            </plan_json>

            Early Life and Music Background Ruand started playing music at a young age, around 10 or 11, initially with the keyboard or piano. He later transitioned to guitar and was interested in flamenco music, but was not accepted into a prestigious program due to his skill level at 16. His father then enrolled him in a sound engineering school, where he developed technical skills in music and sound.

            Education and Career Ruand pursued a degree in performance with a focus on flamenco guitar, but ultimately shifted to sound design for film and television. He was accepted into the Royal College of Arts for his master's degree and worked on various film and game projects.
            """
        )

        self.assertEqual(result["format_name"], "Biography Summary")
        self.assertNotIn("plan_json", result["summary"])
        self.assertEqual(
            result["summary"],
            "## Early Life and Music Background\n"
            "\n"
            "Ruand started playing music at a young age, around 10 or 11, "
            "initially with the keyboard or piano. He later transitioned to "
            "guitar and was interested in flamenco music, but was not accepted "
            "into a prestigious program due to his skill level at 16. His father "
            "then enrolled him in a sound engineering school, where he developed "
            "technical skills in music and sound.\n\n"
            "## Education and Career\n"
            "\n"
            "Ruand pursued a degree in performance with a focus on flamenco guitar, "
            "but ultimately shifted to sound design for film and television. He "
            "was accepted into the Royal College of Arts for his master's degree "
            "and worked on various film and game projects.",
        )

    def test_adaptive_summary_parser_accepts_fenced_json(self):
        result = parse_adaptive_summary_response(
            """```json
{
  "plan": {"format_name": "Meeting Notes", "sections": ["Decisions"]},
  "summary_markdown": "## Decisions\\n- Ship it."
}
```"""
        )

        self.assertEqual(result["format_name"], "Meeting Notes")
        self.assertEqual(result["summary"], "## Decisions\n\n- Ship it.")

    def test_adaptive_summary_parser_extracts_malformed_json_like_response(self):
        result = parse_adaptive_summary_response(
            """{ "plan": { "recording_type": "interview", "format_name": "Artist Background", "audience": "general audience", "sections": ["Early Life and Music Background", "Education and Career"], "style": "informal and conversational", "rationale": "The format is chosen to reflect the informal tone of the interview" }, "summary_markdown": "# Early Life and Music Background Ruand started playing music at a young age, around ten or eleven, initially with keyboard or piano and later with guitar. He wanted to pursue flamenco music, but was rejected due to not being good enough. His dad then enrolled him in a sound engineering school, where he developed technical skills in music and sound.

Education and Career
Ruand went on to study sound design at the Royal College of Arts, National Film and Television Institute, after which he worked on various film and game projects. He is now in the field of sound design." }"""
        )

        self.assertEqual(result["format_name"], "Artist Background")
        self.assertEqual(
            result["summary"],
            "## Early Life and Music Background\n"
            "\n"
            "Ruand started playing music at a young age, around ten or eleven, "
            "initially with keyboard or piano and later with guitar. He wanted "
            "to pursue flamenco music, but was rejected due to not being good "
            "enough. His dad then enrolled him in a sound engineering school, "
            "where he developed technical skills in music and sound.\n\n"
            "## Education and Career\n"
            "\n"
            "Ruand went on to study sound design at the Royal College of Arts, "
            "National Film and Television Institute, after which he worked on "
            "various film and game projects. He is now in the field of sound design.",
        )

    def test_adaptive_summary_parser_converts_bold_heading_lines(self):
        result = parse_adaptive_summary_response(
            """
            <plan_json>
            {"format_name": "Conversation Summary", "sections": ["Conversation Details", "Snack and Drink Options"]}
            </plan_json>
            <summary_markdown>
            **Conversation Details**
            Wikan and Indira discuss snacks. Wikan says **thank you** before asking about options.

            * Indira agrees to get a snack.

            **Snack and Drink Options**
            * Indira decides to buy water.
            </summary_markdown>
            """
        )

        self.assertEqual(
            result["summary"],
            "## Conversation Details\n\n"
            "Wikan and Indira discuss snacks. Wikan says **thank you** before asking about options.\n\n"
            "* Indira agrees to get a snack.\n\n"
            "## Snack and Drink Options\n\n"
            "* Indira decides to buy water.",
        )

    def test_adaptive_summary_parser_falls_back_to_raw_summary(self):
        result = parse_adaptive_summary_response("Plain fallback summary.")

        self.assertEqual(result["format_name"], "General Summary")
        self.assertEqual(result["summary"], "Plain fallback summary.")


if __name__ == "__main__":
    unittest.main()
