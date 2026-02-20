import random

prompt1 = (
    "You are a professional news editor.\n"
    "Write a concise and coherent summary of the following conversation.\n"
    "Focus on the key facts and avoid unnecessary details.\n\n"
)
prompt2 = (
    "Summarize the following conversation in 3–5 sentences.\n"
    "The summary should be factual, concise, and written in a neutral tone.\n\n"
)
prompt3 = (
    "You are a news editor.\n"
    "Write a concise news-style summary of the following broadcast conversation.\n"
    "Include the key events and important statements.\n"
    "Do not repeat information.\n\n"
)

PROMPT_VARIATIONS = [prompt1, prompt2, prompt3]

def build_prompt(transcript: str) -> str:
    """Helper to prompt format the input for MediaSum"""
    instruction = random.choice(PROMPT_VARIATIONS)
    return f"{instruction}\n\nTranscript:\n{transcript}\n\nSummary:"