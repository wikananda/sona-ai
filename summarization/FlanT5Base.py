import os
import sys
from pathlib import Path
from typing import Optional, Union, Dict

# Add project root to sys.path to allow importing from utils
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import yaml
import torch
from datasets import load_dataset, load_from_disk, Features, Value, Sequence
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import numpy as np
import evaluate
from utils.utils import load_config, filter_training_args
from .summarization_dataset import SummarizationDataset

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))

class FlanT5Base:
    def __init__(self, config: Union[str, Dict], use_pretrained: bool = True, device: Optional[str] = None):
        self.config = load_config(config)
        self.project_root = Path(__file__).parent.parent
        self.device = self._get_device(device or self.config['model']['device'])
        self.tokenizer = self._load_tokenizer()
        self.model = self._get_lora_model(use_pretrained)

    def _get_device(self, device):
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device

    def _load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.config['model']['model_name'], 
            cache_dir=str(self.project_root / self.config['cp_dir']['hf_cache'])
        )
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _load_base_model(self):
        model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config['model']['model_name'], 
            cache_dir=str(self.project_root / self.config['cp_dir']['hf_cache'])
        )
        model.config.pad_token_id = self.tokenizer.pad_token_id
        return model

    def _get_lora_model(self, use_pretrained: bool = True):
        base_model = self._load_base_model()
        cp_dir = self.project_root / self.config['model']['cp_dir']
        
        # Check if LoRA adapters already exist
        if use_pretrained and os.path.exists(cp_dir) and os.path.exists(cp_dir / "adapter_config.json"):
            print(f"Using pretrained LoRA adapters from {cp_dir}...")
            model = PeftModel.from_pretrained(base_model, str(cp_dir), is_trainable=True)
        else:
            print(f"Applying new LoRA configuration...")
            lora_config = LoraConfig(
                r=self.config['lora']['r'],
                lora_alpha=self.config['lora']['lora_alpha'],
                target_modules=self.config['lora']['target_modules'],
                lora_dropout=self.config['lora']['lora_dropout'],
                bias=self.config['lora']['bias'],
                task_type=TaskType.SEQ_2_SEQ_LM
            )
            model = get_peft_model(base_model, lora_config)\
                
        if self.config['seq2seq_args']['gradient_checkpointing']:
            model.enable_input_require_grads()
        return model

    def get_data_collator(self):
        return DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
        )
    
    def compute_metrics(self, eval_preds):
        """
        ROUGE compute metrics evaluation
        """
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        # Replace -100s used for padding
        preds = np.where(preds != -100, preds, self.tokenizer.pad_token_id)
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Strip whitespace (ROUGE is sensitive)
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]

        # Compute ROUGE scores
        rouge = evaluate.load("rouge")
        rouge_results = rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True
        )
        return {
            "rouge1": rouge_results["rouge1"],
            "rouge2": rouge_results["rouge2"],
            "rougeL": rouge_results["rougeL"],
            "rougeLsum": rouge_results["rougeLsum"],
        }