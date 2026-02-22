import os
import sys
from pathlib import Path
from typing import Optional, Union, Dict
import gc

# Add project root to sys.path to allow importing from utils
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import yaml
import torch
from datasets import load_dataset, load_from_disk, Features, Value, Sequence
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    DataCollatorForLanguageModeling,
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


class SummarizationBase:
    """
    Generic base class for HuggingFace summarization models.
    Supports both Seq2Seq models (T5, BART, Pegasus, etc.)
    and CausalLM models (LLaMA, Mistral, GPT-2, etc.) via LoRA.
    """

    TASK_TYPE_SEQ2SEQ = "seq2seq"
    TASK_TYPE_CAUSAL = "causal"

    def __init__(
        self,
        config: Union[str, Dict],
        base_model: bool = False,
        use_pretrained: bool = True,
        device: Optional[str] = None,
    ):
        self.config = load_config(config)
        self.project_root = Path(__file__).parent.parent
        self.device = self._get_device(device or self.config['model']['device'])
        self.task_type = self._detect_task_type()
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_base_model() if base_model else self._get_lora_model(use_pretrained)
        print(f"Using device: {self.device}")
        print(f"Task type: {self.task_type}")

    
    # DEVICE HELPERS

    def _get_device(self, device: str) -> str:
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device


    def _detect_task_type(self) -> str:
        """
        Resolve the task type from config or by probing the model class.
        Returns either TASK_TYPE_SEQ2SEQ or TASK_TYPE_CAUSAL.
        """
        cfg_task_type = self.config['model'].get('task_type', 'auto').lower()

        if cfg_task_type == self.TASK_TYPE_SEQ2SEQ:
            return self.TASK_TYPE_SEQ2SEQ
        elif cfg_task_type == self.TASK_TYPE_CAUSAL:
            return self.TASK_TYPE_CAUSAL
        else:
            # "auto" — try to load as Seq2Seq, fallback to CausalLM
            return self._probe_task_type()

    def _probe_task_type(self) -> str:
        """
        Tries to auto-detect whether the model is Seq2Seq or CausalLM
        by attempting to load with AutoModelForSeq2SeqLM first.
        """
        model_name = self.config['model']['model_name']
        cache_dir = str(self.project_root / self.config['cp_dir']['hf_cache'])
        try:
            from transformers import AutoConfig
            model_cfg = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
            # Seq2Seq models have an encoder config (EncoderDecoderConfig or have is_encoder_decoder=True)
            if getattr(model_cfg, 'is_encoder_decoder', False):
                print(f"Auto-detected task type: seq2seq (model: {model_name})")
                return self.TASK_TYPE_SEQ2SEQ
            else:
                print(f"Auto-detected task type: causal (model: {model_name})")
                return self.TASK_TYPE_CAUSAL
        except Exception as e:
            print(f"Warning: Could not auto-detect task type ({e}). Defaulting to seq2seq.")
            return self.TASK_TYPE_SEQ2SEQ

    
    # MODEL AND TOKENIZER LOADING

    def _load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.config['model']['model_name'],
            cache_dir=str(self.project_root / self.config['cp_dir']['hf_cache'])
        )
        # Both Seq2Seq and CausalLM need a pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # CausalLM: left-pad so the model attends correctly during batched generation
        if self.task_type == self.TASK_TYPE_CAUSAL:
            tokenizer.padding_side = "left"

        return tokenizer

    def _load_base_model(self):
        model_name = self.config['model']['model_name']
        cache_dir = str(self.project_root / self.config['cp_dir']['hf_cache'])

        if self.task_type == self.TASK_TYPE_SEQ2SEQ:
            print(f"Loading Seq2Seq base model from {model_name}...")
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir)
        else:
            print(f"Loading CausalLM base model from {model_name}...")
            model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)

        model.config.pad_token_id = self.tokenizer.pad_token_id
        return model

    def _get_lora_model(self, use_pretrained: bool = True):
        base_model = self._load_base_model()
        cp_dir = self.project_root / self.config['model']['cp_dir']

        # Map task type to PEFT TaskType enum
        peft_task_type = (
            TaskType.SEQ_2_SEQ_LM
            if self.task_type == self.TASK_TYPE_SEQ2SEQ
            else TaskType.CAUSAL_LM
        )

        if use_pretrained and os.path.exists(cp_dir) and os.path.exists(cp_dir / "adapter_config.json"):
            print(f"Using pretrained LoRA adapters from {cp_dir}...")
            model = PeftModel.from_pretrained(base_model, str(cp_dir), is_trainable=True)
        else:
            print(f"Applying new LoRA configuration (task_type={peft_task_type.value})...")
            lora_config = LoraConfig(
                r=self.config['lora']['r'],
                lora_alpha=self.config['lora']['lora_alpha'],
                target_modules=self.config['lora']['target_modules'],
                lora_dropout=self.config['lora']['lora_dropout'],
                bias=self.config['lora']['bias'],
                task_type=peft_task_type
            )
            model = get_peft_model(base_model, lora_config)

        if self.config.get('seq2seq_args', self.config.get('training_args', {})).get('gradient_checkpointing', False):
            model.enable_input_require_grads()

        return model


    # UTILS

    def cleanup_models(self):
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def get_data_collator(self):
        if self.task_type == self.TASK_TYPE_SEQ2SEQ:
            return DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer,
                model=self.model,
            )
        else:
            # CausalLM: standard language modeling collator (labels = input_ids shifted by model)
            return DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,  # causal LM, not masked LM
            )

    def compute_metrics(self, eval_preds):
        """
        ROUGE compute metrics. Works for both Seq2Seq and CausalLM evaluation.
        """
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        # Replace -100s used for padding
        preds = np.where(preds != -100, preds, self.tokenizer.pad_token_id)
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]

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
