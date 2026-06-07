import threading
from typing import Optional

from sona_ai.core import resolve_device, validate_device_available
from sona_ai.summarization.prompts import (
    build_one_call_adaptive_summary_prompt,
    parse_adaptive_summary_response,
)


SUPPORTED_LOCAL_LLM_MODELS = {
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
        mode: str = "local",
        byok: Optional[dict] = None,
    ) -> str:
        if mode == "byok":
            return self._summarize_byok(
                text=text,
                prompt=prompt,
                max_length=max_length,
                byok=byok,
            )

        model_name = self._normalize_model(model or self.config)
        summarizer = self._get_summarizer(model_name, device)
        input_limit = (
            max_length
            if max_length is not None
            else self._model_input_limit(model_name)
        )
        return summarizer.generate(text, prompt, max_length=input_limit)

    def summarize_prompt(
        self,
        prompt: str,
        max_length: Optional[int] = None,
        model: Optional[str] = None,
        device: Optional[str] = None,
        mode: str = "local",
        byok: Optional[dict] = None,
    ) -> str:
        if mode == "byok":
            return self._summarize_byok_prompt(
                prompt=prompt,
                max_length=max_length,
                byok=byok,
            )

        model_name = self._normalize_model(model or self.config)
        summarizer = self._get_summarizer(model_name, device)
        input_limit = (
            max_length
            if max_length is not None
            else self._model_input_limit(model_name)
        )
        return summarizer.generate_from_prompt(prompt, max_length=input_limit)

    def summarize_adaptive(
        self,
        text: str,
        prompt: Optional[str] = None,
        max_length: Optional[int] = None,
        model: Optional[str] = None,
        device: Optional[str] = None,
        mode: str = "local",
        byok: Optional[dict] = None,
    ) -> dict:
        adaptive_prompt = build_one_call_adaptive_summary_prompt(text, prompt)
        raw_response = self.summarize_prompt(
            prompt=adaptive_prompt,
            max_length=max_length,
            model=model,
            device=device,
            mode=mode,
            byok=byok,
        )

        return parse_adaptive_summary_response(raw_response)


    def _summarize_byok(
        self,
        text: str,
        prompt: Optional[str],
        max_length: Optional[int],
        byok: Optional[dict],
    ) -> str:
        if byok is None:
            raise ValueError("BYOK settings are required")

        api_key = (byok.get("api_key") or "").strip()
        model = (byok.get("model") or "").strip()
        provider = (byok.get("provider", "") or "").strip()
        base_url = (byok.get("base_url") or "").strip()

        if not api_key:
            raise ValueError("BYOK: api_key is required")
        if not model:
            raise ValueError("BYOK: model is required")

        from sona_ai.summarization import OpenAICompatibleSummarizer

        summarizer = OpenAICompatibleSummarizer()
        return summarizer.generate(
            text=text,
            prompt=prompt,
            api_key=api_key,
            model=model,
            provider=provider,
            base_url=base_url,
            max_length=max_length or 2048,
            max_tokens=self._byok_output_limit(),
        )

    def _summarize_byok_prompt(
        self,
        prompt: str,
        max_length: Optional[int],
        byok: Optional[dict],
    ) -> str:
        if byok is None:
            raise ValueError("BYOK settings are required")

        api_key = (byok.get("api_key") or "").strip()
        model = (byok.get("model") or "").strip()
        provider = (byok.get("provider", "") or "").strip()
        base_url = (byok.get("base_url") or "").strip()

        if not api_key:
            raise ValueError("BYOK: api_key is required")
        if not model:
            raise ValueError("BYOK: model is required")

        from sona_ai.summarization import OpenAICompatibleSummarizer

        summarizer = OpenAICompatibleSummarizer()
        return summarizer.generate_from_prompt(
            prompt=prompt,
            api_key=api_key,
            model=model,
            provider=provider,
            base_url=base_url,
            max_length=max_length or 2048,
            max_tokens=self._byok_output_limit(),
        )

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
        if model_name not in SUPPORTED_LOCAL_LLM_MODELS:
            allowed = ", ".join(sorted(SUPPORTED_LOCAL_LLM_MODELS))
            raise ValueError(f"Unsupported summarization model: {model}. Use one of: {allowed}")
        return SUPPORTED_LOCAL_LLM_MODELS[model_name]

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

    def _byok_output_limit(self) -> int:
        if self.max_new_tokens is not None:
            return self.max_new_tokens
        return 1024
