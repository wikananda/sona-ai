from .FlanT5Base import FlanT5Base
from .prompt import build_prompt

import torch
import numpy as np
import os
import json
import gc

class FlanT5Inferencer(FlanT5Base):
    def __init__(
        self,
        config_name: str = "flan-t5",
        use_pretrained: bool = True,
        device: str = "auto",
        max_new_tokens: int = 256,
        num_beams: int = 4,
    ):
        super().__init__(config_name, use_pretrained=use_pretrained, device=device)
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.model.to(self.device)
        self.model.eval()
        
    def generate(self, text: str, max_length: int = 256) -> str:
        """
        Run summarization script
        """
        prompt = build_prompt(text)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length = max_length,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens = self.max_new_tokens,
                num_beams = self.num_beams,
                early_stopping = True,
            )

        summary = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
        )

        return summary
    
    def cleanup_models(self):
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()