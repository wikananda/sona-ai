import threading
from typing import Optional

from sona_ai.core import resolve_device, validate_device_available


SUPPORTED_SUMMARY_MODELS = {
    "qwen": "qwen",
    "llama": "llama",
    "gemma": "gemma",
}


class SummarizationService:
    def __init__(
        self,
        config: str = "qwen",
        use_pretrained: bool = True,
        device: str = "auto",
        max_new_tokens: int = 256,
        num_beams: int = 4,
    ):
        self.config = self._normalize_model(config)
        self.use_pretrained = use_pretrained
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self._summarizers = {}
        self._lock = threading.Lock()

    def summarize(
        self,
        text: str,
        prompt: Optional[str] = None,
        max_length: int = 2048,
        model: Optional[str] = None,
        device: Optional[str] = None,
    ) -> str:
        summarizer = self._get_summarizer(model, device)
        return summarizer.generate(text, prompt, max_length=max_length)

    def close(self):
        for summarizer in self._summarizers.values():
            summarizer.cleanup_models()
        self._summarizers = {}

    def _get_summarizer(
        self,
        model: Optional[str] = None,
        device: Optional[str] = None,
    ):
        model_name = self._normalize_model(model or self.config)
        device_name = validate_device_available(device or self.device)
        key = self._cache_key(model_name, device_name)
        if key in self._summarizers:
            return self._summarizers[key]

        with self._lock:
            if key in self._summarizers:
                return self._summarizers[key]

            self._summarizers[key] = self._build_summarizer(model_name, device_name)
            return self._summarizers[key]

    def _build_summarizer(self, model_name: str, device: str):
        from sona_ai.core import load_config

        config = load_config(model_name)
        backend = config.get("model", {}).get("backend", "transformers")

        if backend == "gguf":
            from sona_ai.summarization import GGUFLLMSummarizer

            return GGUFLLMSummarizer(
                config=config,
                max_new_tokens=self.max_new_tokens,
                device=device,
            )

        if backend == "gemma4":
            from sona_ai.summarization import Gemma4Summarizer

            return Gemma4Summarizer(
                config=config,
                max_new_tokens=self.max_new_tokens,
                device=device,
            )

        from sona_ai.summarization import LocalLLMSummarizer

        return LocalLLMSummarizer(
            config=config,
            use_pretrained=self.use_pretrained,
            device=resolve_device(device),
            max_new_tokens=self.max_new_tokens,
            num_beams=self.num_beams,
        )

    def _normalize_model(self, model: str) -> str:
        model_name = model.lower().strip()
        if model_name not in SUPPORTED_SUMMARY_MODELS:
            allowed = ", ".join(sorted(SUPPORTED_SUMMARY_MODELS))
            raise ValueError(f"Unsupported summarization model: {model}. Use one of: {allowed}")
        return SUPPORTED_SUMMARY_MODELS[model_name]

    def _cache_key(self, model: str, device: str) -> tuple[str, str]:
        return (model, resolve_device(device))
