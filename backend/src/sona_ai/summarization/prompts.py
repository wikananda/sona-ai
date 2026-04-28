import random
from typing import Optional


PROMPT_VARIATIONS = [
    (
        "You are a professional news editor.\n"
        "Write a concise and coherent summary of the following conversation.\n"
        "Focus on the key facts and avoid unnecessary details.\n\n"
    ),
    (
        "Summarize the following conversation in 3-5 sentences.\n"
        "The summary should be factual, concise, and written in a neutral tone.\n\n"
    ),
    (
        "You are a news editor.\n"
        "Write a concise news-style summary of the following broadcast conversation.\n"
        "Include the key events and important statements.\n"
        "Do not repeat information.\n\n"
    ),
]


def build_prompt(transcript: str, prompt: Optional[str] = None) -> str:
    instruction = prompt if prompt else random.choice(PROMPT_VARIATIONS)
    return f"{instruction}\n\nTranscript:\n{transcript}\n\nSummary:"

