import os
from pathlib import Path
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
from peft import LoraConfig, get_peft_model, TaskType
from utils.utils import load_config, filter_training_args
from summarization_dataset import SummarizationDataset

# Load configuration
script_dir = Path(__file__).parent
project_root = script_dir.parent
config = load_config('flan-t5')

MODEL_NAME = config['model']['model_name']
MAX_INPUT_LENGTH = config['model']['max_input_length']
MAX_TARGET_LENGTH = config['model']['max_target_length']
HF_CACHE = project_root / config['cp_dir']['hf_cache']
DATASET_DIR = project_root / config['dataset']['dataset_dir']
OUTPUT_DIR = project_root / config['model']['output_dir']

# Load model and tokenizer
# use_fast=False is more reliable for T5 models to avoid normalization errors
print(f"Loading base model and tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE)
# T5 has no padding token by default. Use EOS token as padding token
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id
# print(model)

def compute_metrics(eval_preds):
    """
    ROUGE compute metrics evaluation
    """
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    # Replace -100s used for padding
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

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

# LoRA Configuration
print("Applying LoRA configuration...")
lora_config = LoraConfig(
    r=config['lora']['r'], # Rank: Determines the number of trainable parameters
    lora_alpha=config['lora']['lora_alpha'], # Scaling factor
    target_modules=config['lora']['target_modules'], # Target the query and value layers of attention.
    lora_dropout=config['lora']['lora_dropout'],
    bias=config['lora']['bias'],
    task_type=TaskType.SEQ_2_SEQ_LM
)
model = get_peft_model(model, lora_config)
if config['seq2seq_args']['gradient_checkpointing']:
    model.enable_input_require_grads() # Required for gradient checkpointing
model.print_trainable_parameters()

# Load Dataset
dataset = SummarizationDataset(
    tokenizer=tokenizer,
    name="mediasum",
    source='local',
    seed=42,
    train_size=5000,
    val_size=500,
    test_size=500
)
tokenized_dataset = dataset.load_and_prepare()

# Training Arguments
seq2seq_args = filter_training_args(config['seq2seq_args'])
training_args = Seq2SeqTrainingArguments(
    output_dir=str(OUTPUT_DIR),
    **seq2seq_args
)

# data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['val'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

print("Starting LoRA fine-tuning...")
trainer.train()

print(f"Saving LoRA adapters to {OUTPUT_DIR}")
model.save_pretrained(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))
