from typing import Optional

from sona_ai.summarization import LocalLLMSummarizer


class SummarizationService:
    def __init__(self, summarizer: LocalLLMSummarizer):
        self.summarizer = summarizer

    def summarize(self, text: str, prompt: Optional[str] = None, max_length: int = 2048) -> str:
        return self.summarizer.generate(text, prompt, max_length=max_length)

    def close(self):
        self.summarizer.cleanup_models()

