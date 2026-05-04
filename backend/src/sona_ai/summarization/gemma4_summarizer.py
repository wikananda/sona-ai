import gc
from typing import Dict, Optional, Union

import torch
from packaging.version import Version

from sona_ai.core import PROJECT_ROOT, load_config, write_json
from sona_ai.summarization.prompts import build_prompt


MIN_TRANSFORMERS_VERSION = Version("5.5.0")


class Gemma4Summarizer:
    def __init__(
        self,
        config: Union[str, Dict] = "gemma",
        max_new_tokens: int = 256,
        write_outputs: bool = False,
    ):
        self.config = load_config(config)
        self.max_new_tokens = max_new_tokens
        self.write_outputs = write_outputs
        self.device = self._resolve_device(self.config["model"].get("device", "auto"))
        self.processor = None
        self.model = None

        self._validate_transformers_version()
        self.processor = self._load_processor()
        self.model = self._load_model()
        if self.device != "auto":
            self.model.to(self.device)
        self.model.eval()

    def _cache_dir(self) -> str:
        return str(PROJECT_ROOT / self.config["cp_dir"]["hf_cache"])

    def _validate_transformers_version(self) -> None:
        import transformers

        current = Version(transformers.__version__.split("+", 1)[0])
        if current < MIN_TRANSFORMERS_VERSION:
            raise RuntimeError(
                "google/gemma-4-E2B-it requires transformers>=5.5.0. "
                f"Current version is {transformers.__version__}. Run "
                "`pip install --upgrade 'transformers>=5.5.0'` in the sona-ai "
                "environment, then restart the backend."
            )

    def _load_processor(self):
        from transformers import AutoProcessor

        return AutoProcessor.from_pretrained(
            self.config["model"]["model_name"],
            cache_dir=self._cache_dir(),
        )

    def _load_model(self):
        from transformers import AutoModelForCausalLM

        kwargs = {
            "cache_dir": self._cache_dir(),
            "dtype": self._torch_dtype(),
        }
        if self.device == "auto":
            kwargs["device_map"] = "auto"

        return AutoModelForCausalLM.from_pretrained(
            self.config["model"]["model_name"],
            **kwargs,
        )

    def generate(self, text: str, prompt: Optional[str] = None, max_length: int = 2048) -> str:
        formatted_prompt = build_prompt(text, prompt)
        messages = [
            {
                "role": "system",
                "content": "You summarize transcripts accurately and concisely.",
            },
            {
                "role": "user",
                "content": formatted_prompt[:max_length * 4],
            },
        ]
        chat_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.processor(text=chat_text, return_tensors="pt").to(self._input_device())
        input_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        response = self.processor.decode(
            outputs[0][input_len:],
            skip_special_tokens=False,
        )
        summary = self._parse_response(response)

        if self.write_outputs:
            output_path = PROJECT_ROOT / "outputs" / "summarization" / "summary.json"
            write_json(output_path, {"text": text, "summary": summary})

        return summary

    def _parse_response(self, response: str) -> str:
        if hasattr(self.processor, "parse_response"):
            parsed = self.processor.parse_response(response)
            if isinstance(parsed, dict):
                return str(parsed.get("answer") or parsed.get("response") or response).strip()
            if isinstance(parsed, str):
                return parsed.strip()
        return response.strip()

    def _resolve_device(self, device: str) -> str:
        if device != "auto":
            return device
        if torch.cuda.is_available():
            return "auto"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _torch_dtype(self):
        if self.device == "cpu":
            return torch.float32
        return "auto"

    def _input_device(self):
        if self.device != "auto":
            return torch.device(self.device)
        return self.model.device

    def cleanup_models(self):
        if self.model is not None:
            del self.model
            self.model = None
        self.processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
