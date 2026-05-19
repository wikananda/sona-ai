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
        max_new_tokens: Optional[int] = None,
        num_beams: Optional[int] = None,
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
        max_length: Optional[int] = None,
        model: Optional[str] = None,
        device: Optional[str] = None,
    ) -> str:
        model_name = self._normalize_model(model or self.config)
        summarizer = self._get_summarizer(model_name, device)
        input_limit = (
            max_length
            if max_length is not None
            else self._model_input_limit(model_name)
        )
        return summarizer.generate(text, prompt, max_length=input_limit)

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
        max_new_tokens = self._model_output_limit(config)
        num_beams = self._model_num_beams(config)

        if backend == "gguf":
            from sona_ai.summarization import GGUFLLMSummarizer

            return GGUFLLMSummarizer(
                config=config,
                max_new_tokens=max_new_tokens,
                device=device,
            )

        if backend == "gemma4":
            from sona_ai.summarization import Gemma4Summarizer

            return Gemma4Summarizer(
                config=config,
                max_new_tokens=max_new_tokens,
                device=device,
            )

        from sona_ai.summarization import LocalLLMSummarizer

        return LocalLLMSummarizer(
            config=config,
            use_pretrained=self.use_pretrained,
            device=resolve_device(device),
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )

    def _normalize_model(self, model: str) -> str:
        model_name = model.lower().strip()
        if model_name not in SUPPORTED_SUMMARY_MODELS:
            allowed = ", ".join(sorted(SUPPORTED_SUMMARY_MODELS))
            raise ValueError(f"Unsupported summarization model: {model}. Use one of: {allowed}")
        return SUPPORTED_SUMMARY_MODELS[model_name]

    def _cache_key(self, model: str, device: str) -> tuple[str, str]:
        return (model, resolve_device(device))

    def _model_input_limit(self, model_name: str) -> int:
        from sona_ai.core import load_config

        config = load_config(model_name)
        return int(config.get("limits", {}).get("max_input_length", 2048))

    def _model_output_limit(self, config: dict) -> int:
        configured = self.max_new_tokens
        if configured is not None:
            return configured
        return int(config.get("limits", {}).get("max_output_tokens", 256))

    def _model_num_beams(self, config: dict) -> int:
        configured = self.num_beams
        if configured is not None:
            return configured
        return int(config.get("generation", {}).get("num_beams", 4))
