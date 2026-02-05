import os
from pathlib import Path
from datasets import load_dataset, load_from_disk, Features, Value, Sequence
from transformers import AutoTokenizer
from prompt import build_prompt

from utils.utils import load_config

class SummarizationDataset:
    def __init__(
        self,
        tokenizer,
        name="mediasum",
        source=None,
        seed=42,
        train_size=5000,
        val_size=500,
        test_size=500,
    ):
        self.project_root = Path(__file__).parent.parent
        self.config = load_config('flan-t5')
        self.tokenizer = tokenizer
        self.name = name
        self.source = source or self.config['dataset'].get('dataset_source', 'local')
        self.seed = seed
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size

    def load_raw(self):
        if self.source == "local":
            if self.name == "mediasum":
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
            # Online data loading from Hugging Face Hub
            dataset = load_dataset(self.config['dataset']['dataset_name'])
        # Safely select subsets
        for split, size in [('train', self.train_size), ('val', self.val_size), ('test', self.test_size)]:
            if size is not None and size != -1:
                actual_size = min(size, len(dataset[split]))
                dataset[split] = dataset[split].shuffle(seed=self.seed).select(range(actual_size))
            else:
                # Still shuffle even if selecting all, if a seed is provided
                dataset[split] = dataset[split].shuffle(seed=self.seed)

        print(f"Train size: {len(dataset['train'])} | Val size: {len(dataset['val'])} | Test size: {len(dataset['test'])}")
        return dataset

    def load_and_prepare(self):
        dataset = self.load_raw()
        tokenized_dir = str(self.project_root / self.config['dataset']['tokenized_dir'] / f"{self.name}_tokenized_{self.config['model']['model_name'].replace('/', '_')}")
        try:
            print("Loading processed dataset from disk...")
            tokenized_dataset = load_from_disk(tokenized_dir)
        except FileNotFoundError:
            print("Failed to load dataset from disk. Tokenizing dataset...")
            tokenized_dataset = dataset.map(
                self.preprocess,
                batched=True,
                remove_columns=dataset['train'].column_names
            )
            tokenized_dataset.save_to_disk(str(self.project_root / tokenized_dir))
        return tokenized_dataset
        
    def preprocess(self, batch):
        """Tokenize the batch of data"""
        transcripts = []
        # MediaSum local files use 'speaker' (list) and 'utt' (list)
        for speakers_list, utt_list in zip(batch['speaker'], batch['utt']):
            transcript = ""
            for speakers, utterances in zip(speakers_list, utt_list):
                transcript += f"{speakers}: {utterances}\n"
            transcripts.append(build_prompt(transcript)) # for instruction tuning

        # Tokenize input transcripts
        model_inputs = self.tokenizer(
            transcripts,
            max_length=self.config['model']['max_input_length'],
            truncation=True,
            padding="max_length",
        )

        # Tokenize target summaries
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                batch['summary'],
                max_length=self.config['model']['max_target_length'],
                truncation=True,
                padding="max_length",
            )
        
        # Filter out padding tokens
        labels_ids = labels["input_ids"]
        labels_ids = [
            [lid if lid != self.tokenizer.pad_token_id else -100 for lid in label]
            for label in labels_ids
        ]
        model_inputs["labels"] = labels_ids
        return model_inputs
    