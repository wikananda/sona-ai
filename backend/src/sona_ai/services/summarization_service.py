import threading
from typing import Optional


class SummarizationService:
    def __init__(
        self,
        config: str = "qwen",
        use_pretrained: bool = True,
        device: str = "auto",
        max_new_tokens: int = 256,
        num_beams: int = 4,
    ):
        self.config = config
        self.use_pretrained = use_pretrained
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.summarizer = None
        self._lock = threading.Lock()

    def summarize(self, text: str, prompt: Optional[str] = None, max_length: int = 2048) -> str:
        summarizer = self._get_summarizer()
        return summarizer.generate(text, prompt, max_length=max_length)

    def close(self):
        if self.summarizer is not None:
            self.summarizer.cleanup_models()
            self.summarizer = None

    def _get_summarizer(self):
        if self.summarizer is not None:
            return self.summarizer

        with self._lock:
            if self.summarizer is not None:
                return self.summarizer

            self.summarizer = self._build_summarizer()
            return self.summarizer

    def _build_summarizer(self):
        from sona_ai.core import load_config

        config = load_config(self.config)
        backend = config.get("model", {}).get("backend", "transformers")

        if backend == "gguf":
            from sona_ai.summarization import GGUFLLMSummarizer

            return GGUFLLMSummarizer(
                config=config,
                max_new_tokens=self.max_new_tokens,
            )

        from sona_ai.summarization import LocalLLMSummarizer

        return LocalLLMSummarizer(
            config=config,
            use_pretrained=self.use_pretrained,
            device=self.device,
            max_new_tokens=self.max_new_tokens,
            num_beams=self.num_beams,
        )
