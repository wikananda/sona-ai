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
import evaluate
import numpy as np
from inspect import signature

# Load configuration
script_dir = Path(__file__).parent
project_root = script_dir.parent
config_path = project_root / "configs" / "flan-t5.yaml"

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

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

def build_prompt(transcript: str) -> str:
    """Helper to prompt format the input for MediaSum"""
    return (
        "Summarize the following conversation into a concise, well-written summary:\n\n"
        f"{transcript}\n\n"
        "Summary:"
    )

def preprocess(batch):
    """Tokenize the batch of data"""
    transcripts = []
    # MediaSum local files use 'speaker' (list) and 'utt' (list)
    for speakers_list, utt_list in zip(batch['speaker'], batch['utt']):
        transcript = ""
        for speakers, utterances in zip(speakers_list, utt_list):
            transcript += f"{speakers}: {utterances}\n"
        transcripts.append(build_prompt(transcript)) # for instruction tuning

    # Tokenize input transcripts
    model_inputs = tokenizer(
        transcripts,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding="max_length",
    )

    # Tokenize target summaries
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            batch['summary'],
            max_length=MAX_TARGET_LENGTH,
            truncation=True,
            padding="max_length",
        )
    
    # Filter out padding tokens
    labels_ids = labels["input_ids"]
    labels_ids = [
        [lid if lid != tokenizer.pad_token_id else -100 for lid in label]
        for label in labels_ids
    ]
    model_inputs["labels"] = labels_ids
    return model_inputs

# Load Dataset
print("Loading local MediaSum dataset...")
# Define features explicitly to avoid PyArrow schema inference errors (especially on the 'date' field)
features = Features({
    "id": Value("string"),
    "program": Value("string"),
    "date": Value("string"), # treating date as text which is perfect for lLM
    "url": Value("string"),
    "title": Value("string"),
    "summary": Value("string"),
    "utt": Sequence(Value("string")), # specify this is a list of string
    "speaker": Sequence(Value("string"))
})

dataset = load_dataset(
    "json",
    data_files={
        "train": str(DATASET_DIR / config['dataset']['train_data_file']),
        "val": str(DATASET_DIR / config['dataset']['val_data_file']),
        "test": str(DATASET_DIR / config['dataset']['test_data_file'])
    },
    features=features
)
# Optional: Subset for debugging/testing
# print("Selecting subset for testing...")
# dataset['train'] = dataset['train'].shuffle(seed=42).select(range(100))
# dataset['val'] = dataset['val'].shuffle(seed=42).select(range(10))

print("Preprocessing dataset...")
tokenized_dir = str(project_root / config['dataset']['tokenized_dir'] / f"mediasum_tokenized_{MODEL_NAME.replace('/', '_')}")
try:
    print("Loading processed dataset from disk...")
    tokenized_dataset = load_from_disk(tokenized_dir)
except FileNotFoundError:
    print("Failed to load dataset from disk. Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset['train'].column_names
    )
    tokenized_dataset.save_to_disk(str(project_root / tokenized_dir))

def filter_training_args(arg_dict):
    """
    Gather all arguments inside the .yaml file automatically
    """
    # Get the parameters
    sig = signature(Seq2SeqTrainingArguments).parameters
    valid_keys = set(sig.keys())

    filtered_args = {}
    unused = []

    for k, v in arg_dict.items():
        if k in valid_keys:
            filtered_args[k] = v
        else:
            unused.append(k)

    if unused:
        print(f"Warning: Unused arguments: {unused}")
    return filtered_args


# Compute metrics func
rouge = evaluate.load("rouge")
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
