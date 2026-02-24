import os
from pathlib import Path
from datasets import load_dataset, load_from_disk, Features, Value, Sequence
from transformers import AutoTokenizer
from .prompt import build_prompt

from utils.utils import load_config


class SummarizationDataset:
    def __init__(
        self,
        tokenizer,
        config=None,
        task_type: str = "seq2seq",
        name: str = "mediasum",
        source=None,
        seed: int = 42,
        train_size: int = 5000,
        val_size: int = 500,
        test_size: int = 500,
    ):
        self.project_root = Path(__file__).parent.parent
        self.config = load_config(config) if config else load_config('llama')
        self.tokenizer = tokenizer
        self.task_type = task_type
        self.name = name
        self.source = source or self.config['dataset'].get('dataset_source', 'local')
        self.seed = seed
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size

    def load_raw(self):
        """
        Loads the raw dataset from the specified source.
        """
        if self.source == "local":
            if self.name == "mediasum":
                features = Features({
                    "id": Value("string"),
                    "program": Value("string"),
                    "date": Value("string"),
                    "url": Value("string"),
                    "title": Value("string"),
                    "summary": Value("string"),
                    "utt": Sequence(Value("string")),
                    "speaker": Sequence(Value("string"))
                })
                dataset_dir = self.project_root / self.config['dataset']['dataset_dir']
                dataset = load_dataset(
                    "json",
                    data_files={
                        "train": str(dataset_dir / self.config['dataset']['train_data_file']),
                        "val": str(dataset_dir / self.config['dataset']['val_data_file']),
                        "test": str(dataset_dir / self.config['dataset']['test_data_file'])
                    },
                    features=features
                )
            else:
                raise ValueError(f"Unsupported dataset name: {self.name}")
        else:
            dataset = load_dataset(self.config['dataset']['dataset_name'])

        dataset = self._split_sizes(dataset)
        print(f"Train size: {len(dataset['train'])} | Val size: {len(dataset['val'])} | Test size: {len(dataset['test'])}")
        return dataset

    def load_and_prepare(self):
        """
        Loads, preprocesses, and tokenizes the dataset.
        Uses a task_type-specific cache directory to avoid collisions.
        """
        model_name_safe = self.config['model']['model_name'].replace('/', '_')
        tokenized_dir = str(
            self.project_root
            / self.config['dataset']['tokenized_dir']
            / f"{self.name}_{self.task_type}_{model_name_safe}"
        )
        try:
            print("Loading processed dataset from disk...")
            tokenized_dataset = load_from_disk(tokenized_dir)
        except FileNotFoundError:
            print("Failed to load dataset from disk. Tokenizing dataset...")
            dataset = self.load_raw()
            preprocess_fn = (
                self._preprocess_seq2seq
                if self.task_type == "seq2seq"
                else self._preprocess_causal
            )
            tokenized_dataset = dataset.map(
                preprocess_fn,
                batched=True,
                remove_columns=dataset['train'].column_names
            )
            tokenized_dataset.save_to_disk(tokenized_dir)

        tokenized_dataset = self._split_sizes(tokenized_dataset)
        return tokenized_dataset

    def _split_sizes(self, dataset):
        for split, size in [('train', self.train_size), ('val', self.val_size), ('test', self.test_size)]:
            if size is not None and size != -1:
                actual_size = min(size, len(dataset[split]))
                dataset[split] = dataset[split].shuffle(seed=self.seed).select(range(actual_size))
            else:
                dataset[split] = dataset[split].shuffle(seed=self.seed)
        return dataset

    def _build_transcripts(self, batch) -> list:
        """Build formatted transcript strings from speaker/utterance pairs."""
        transcripts = []
        for speakers_list, utt_list in zip(batch['speaker'], batch['utt']):
            transcript = ""
            for speaker, utterance in zip(speakers_list, utt_list):
                transcript += f"{speaker}: {utterance}\n"
            transcripts.append(transcript)
        return transcripts

    def _preprocess_seq2seq(self, batch):
        """
        Seq2Seq preprocessing: separate encoder input (prompt) and decoder target (summary).
        Labels are masked at padding positions (-100).
        """
        transcripts = [build_prompt(t) for t in self._build_transcripts(batch)]

        model_inputs = self.tokenizer(
            transcripts,
            max_length=self.config['model']['max_input_length'],
            truncation=True,
            padding="max_length",
        )

        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                batch['summary'],
                max_length=self.config['model']['max_target_length'],
                truncation=True,
                padding="max_length",
            )

        labels_ids = [
            [lid if lid != self.tokenizer.pad_token_id else -100 for lid in label]
            for label in labels["input_ids"]
        ]
        model_inputs["labels"] = labels_ids
        return model_inputs

    def _preprocess_causal(self, batch):
        """
        CausalLM preprocessing (SFT-style): concatenate prompt + summary into a single
        sequence. Prompt tokens are masked as -100 in labels so the model only learns
        to predict the summary part.
        """
        transcripts = self._build_transcripts(batch)
        input_ids_list, attention_mask_list, labels_list = [], [], []

        for transcript, summary in zip(transcripts, batch['summary']):
            prompt_text = build_prompt(transcript)
            full_text = prompt_text + summary

            prompt_enc = self.tokenizer(
                prompt_text,
                max_length=self.config['model']['max_input_length'],
                truncation=True,
            )
            full_enc = self.tokenizer(
                full_text,
                max_length=self.config['model']['max_input_length'] + self.config['model']['max_target_length'],
                truncation=True,
                padding="max_length",
            )

            prompt_len = len(prompt_enc["input_ids"])
            input_ids = full_enc["input_ids"]
            attention_mask = full_enc["attention_mask"]

            # Mask prompt tokens in labels — model only learns to generate the summary
            labels = [-100] * prompt_len + input_ids[prompt_len:]
            # Mask padding tokens in labels too
            labels = [
                lid if mask == 1 else -100
                for lid, mask in zip(labels, attention_mask)
            ]

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            labels_list.append(labels)

        return {
            "input_ids": input_ids_list,
            "attention_mask": attention_mask_list,
            "labels": labels_list,
        }