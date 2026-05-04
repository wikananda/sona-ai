import gc
from typing import Dict, Optional, Union

from sona_ai.core import PROJECT_ROOT, load_config, write_json
from sona_ai.summarization.prompts import build_prompt


class GGUFLLMSummarizer:
    def __init__(
        self,
        config: Union[str, Dict] = "qwen",
        max_new_tokens: Optional[int] = None,
        write_outputs: bool = False,
    ):
        self.config = load_config(config)
        self.write_outputs = write_outputs
        self.model_config = self.config["model"]
        self.generation_config = self.config.get("generation", {})
        self.max_new_tokens = max_new_tokens or self.generation_config.get("max_new_tokens", 256)

        self.model_path = self._download_model()
        self.model = self._load_model()

    def _cache_dir(self) -> str:
        return str(PROJECT_ROOT / self.config["cp_dir"]["hf_cache"])

    def _download_model(self) -> str:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError(
                "GGUF summarization requires huggingface_hub. "
                "Install it with `pip install -r backend/requirements.txt`."
            ) from exc

        return hf_hub_download(
            repo_id=self.model_config["repo_id"],
            filename=self.model_config["filename"],
            cache_dir=self._cache_dir(),
        )

    def _load_model(self):
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise ImportError(
                "GGUF summarization requires llama-cpp-python. "
                "Install it with `pip install -r backend/requirements.txt`."
            ) from exc

        kwargs = {
            "model_path": self.model_path,
            "n_ctx": self.model_config.get("n_ctx", 8192),
            "n_gpu_layers": self.model_config.get("n_gpu_layers", -1),
            "verbose": self.model_config.get("verbose", False),
        }
        if self.model_config.get("n_threads") is not None:
            kwargs["n_threads"] = self.model_config["n_threads"]

        return Llama(**kwargs)

    def generate(self, text: str, prompt: Optional[str] = None, max_length: int = 2048) -> str:
        formatted_prompt = build_prompt(text, prompt)
        messages = [
            {
                "role": "system",
                "content": "You are a concise assistant that summarizes transcripts accurately.",
            },
            {
                "role": "user",
                "content": formatted_prompt[:max_length * 4],
            },
        ]

        response = self.model.create_chat_completion(
            messages=messages,
            max_tokens=self.max_new_tokens,
            temperature=self.generation_config.get("temperature", 0.2),
            top_p=self.generation_config.get("top_p", 0.9),
            repeat_penalty=self.generation_config.get("repeat_penalty", 1.15),
        )
        summary = response["choices"][0]["message"]["content"].strip()

        if self.write_outputs:
            output_path = PROJECT_ROOT / "outputs" / "summarization" / "summary.json"
            write_json(output_path, {"text": text, "summary": summary})

        return summary

    def cleanup_models(self):
        if getattr(self, "model", None) is not None:
            del self.model
            self.model = None
        gc.collect()
