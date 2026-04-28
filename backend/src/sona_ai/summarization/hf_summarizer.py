import gc
from typing import Dict, Optional, Union

import torch
from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from sona_ai.core import PROJECT_ROOT, load_config, write_json
from sona_ai.summarization.prompts import build_prompt


class LocalLLMSummarizer:
    """
    Runtime summarizer backed by a local Hugging Face model.

    This keeps serving concerns separate from the legacy training package. It can
    load a base model directly, or attach existing PEFT adapters when present.
    """

    TASK_TYPE_SEQ2SEQ = "seq2seq"
    TASK_TYPE_CAUSAL = "causal"

    def __init__(
        self,
        config: Union[str, Dict] = "llama",
        use_pretrained: bool = True,
        device: str = "auto",
        max_new_tokens: int = 256,
        num_beams: int = 4,
        write_outputs: bool = False,
    ):
        self.config = load_config(config)
        self.device = self._resolve_device(device or self.config["model"]["device"])
        self.task_type = self._detect_task_type()
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.write_outputs = write_outputs

        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model(use_pretrained=use_pretrained)
        self.model.to(self.device)
        self.model.eval()

    def _resolve_device(self, device: str) -> str:
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return device

    def _cache_dir(self) -> str:
        return str(PROJECT_ROOT / self.config["cp_dir"]["hf_cache"])

    def _adapter_dir(self):
        return PROJECT_ROOT / self.config["model"]["cp_dir"]

    def _detect_task_type(self) -> str:
        configured_type = self.config["model"].get("task_type", "auto").lower()
        if configured_type in {self.TASK_TYPE_SEQ2SEQ, self.TASK_TYPE_CAUSAL}:
            return configured_type

        model_config = AutoConfig.from_pretrained(
            self.config["model"]["model_name"],
            cache_dir=self._cache_dir(),
        )
        return (
            self.TASK_TYPE_SEQ2SEQ
            if getattr(model_config, "is_encoder_decoder", False)
            else self.TASK_TYPE_CAUSAL
        )

    def _load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.config["model"]["model_name"],
            cache_dir=self._cache_dir(),
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if self.task_type == self.TASK_TYPE_CAUSAL:
            tokenizer.padding_side = "left"
        return tokenizer

    def _load_base_model(self):
        model_name = self.config["model"]["model_name"]
        cache_dir = self._cache_dir()

        if self.task_type == self.TASK_TYPE_SEQ2SEQ:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)

        model.config.pad_token_id = self.tokenizer.pad_token_id
        return model

    def _load_model(self, use_pretrained: bool):
        model = self._load_base_model()
        adapter_dir = self._adapter_dir()

        if use_pretrained and (adapter_dir / "adapter_config.json").exists():
            return PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)

        return model

    def generate(self, text: str, prompt: Optional[str] = None, max_length: int = 2048) -> str:
        formatted_prompt = build_prompt(text, prompt)
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
                no_repeat_ngram_size=3,
                repetition_penalty=1.2,
                early_stopping=True,
            )

        if self.task_type == self.TASK_TYPE_CAUSAL:
            input_length = inputs.input_ids.shape[1]
            generated_ids = output_ids[0][input_length:]
        else:
            generated_ids = output_ids[0]

        summary = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        if self.write_outputs:
            output_path = PROJECT_ROOT / "outputs" / "summarization" / "summary.json"
            write_json(output_path, {"text": text, "summary": summary})

        return summary

    def cleanup_models(self):
        if getattr(self, "model", None) is not None:
            del self.model
            self.model = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

