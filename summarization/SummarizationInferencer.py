from typing import Union, Dict, Optional
from .SummarizationBase import SummarizationBase
from .prompt import build_prompt

import torch
import numpy as np
import os
import json


class SummarizationInferencer(SummarizationBase):
    """
    Inference class for summarization models.
    Supports both Seq2Seq (encoder-decoder) and CausalLM (decoder-only) models.
    """

    def __init__(
        self,
        config: Union[str, Dict] = "llama",
        base_model: bool = False,
        use_pretrained: bool = True,
        device: str = "auto",
        max_new_tokens: int = 256,
        num_beams: int = 4,
    ):
        super().__init__(config, base_model, use_pretrained, device)
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.model.to(self.device)
        self.model.eval()

    def generate(self, text: str, prompt: Optional[str] = None, max_length: int = 256) -> str:
        """
        Run summarization on a transcript.

        Args:
            text: the transcript to summarize
            prompt: optional custom instruction prompt
            max_length: max token length for the input

        Returns:
            str: the generated summary
        """
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
            # CausalLM echoes the input — strip it by slicing only the new tokens
            input_length = inputs.input_ids.shape[1]
            new_token_ids = output_ids[0][input_length:]
            summary = self.tokenizer.decode(new_token_ids, skip_special_tokens=True)
        else:
            # Seq2Seq decoder produces only the target sequence
            summary = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return summary
