from typing import Union, Dict, Optional
from .SummarizationBase import SummarizationBase
from .summarization_dataset import SummarizationDataset
from utils.utils import filter_training_args

from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    Trainer,
    TrainingArguments,
)
import numpy as np
import matplotlib.pyplot as plt
import json
import os


class SummarizationTrainer(SummarizationBase):
    """
    Fine-tuning, testing, and evaluation class for any HuggingFace summarization model.
    Supports both Seq2Seq (T5, BART, etc.) and CausalLM (LLaMA, GPT-2, etc.) via LoRA.
    """

    def __init__(self, config: Union[str, Dict] = "flan-t5"):
        """
        config: str or dict, name of the config, path to YAML, or the config dict.
        """
        super().__init__(config)
        self.tokenized_dataset = None
        self.trainer = None
        # Resolve training args key: seq2seq_args (Seq2Seq) or training_args (CausalLM)
        self.args = self._resolve_training_args()
        self.model.print_trainable_parameters()

    def _resolve_training_args(self) -> dict:
        """
        Returns filtered training args dict based on task type.
        Seq2Seq uses 'seq2seq_args', CausalLM falls back to 'training_args'.
        """
        if self.task_type == self.TASK_TYPE_SEQ2SEQ:
            raw_args = self.config.get('seq2seq_args', self.config.get('training_args', {}))
            return filter_training_args(raw_args, task_type=self.TASK_TYPE_SEQ2SEQ)
        else:
            raw_args = self.config.get('training_args', self.config.get('seq2seq_args', {}))
            return filter_training_args(raw_args, task_type=self.TASK_TYPE_CAUSAL)

    def prepare_data_and_trainer(self, training_args_dict: Optional[Dict] = None, **kwargs):
        """
        Loads and tokenizes the dataset, then builds the trainer.
        training_args_dict: optional dict to override/extend config training arguments.
        kwargs: override train_size, val_size, test_size, etc.
        """
        print("Preparing dataset...")
        ds_manager = SummarizationDataset(
            tokenizer=self.tokenizer,
            config=self.config,
            task_type=self.task_type,
            **kwargs
        )
        self.tokenized_dataset = ds_manager.load_and_prepare()
        self.trainer = self._get_trainer(training_args_dict)

    def _get_trainer(self, training_args_dict: Optional[Dict] = None):
        """
        Returns a Seq2SeqTrainer for Seq2Seq models, or a standard Trainer for CausalLM.
        """
        if self.tokenized_dataset is None:
            raise ValueError("Dataset not prepared. Call prepare_data_and_trainer() first.")

        final_args = self.args.copy()
        if training_args_dict:
            print("Merging custom training arguments...")
            custom_args = filter_training_args(training_args_dict, task_type=self.task_type)
            final_args.update(custom_args)

        if self.task_type == self.TASK_TYPE_SEQ2SEQ:
            training_args = Seq2SeqTrainingArguments(**final_args)
            return Seq2SeqTrainer(
                model=self.model,
                args=training_args,
                train_dataset=self.tokenized_dataset['train'],
                eval_dataset=self.tokenized_dataset['val'],
                tokenizer=self.tokenizer,
                data_collator=self.get_data_collator(),
                compute_metrics=self.compute_metrics,
            )
        else:
            training_args = TrainingArguments(**final_args)
            return Trainer(
                model=self.model,
                args=training_args,
                train_dataset=self.tokenized_dataset['train'],
                eval_dataset=self.tokenized_dataset['val'],
                tokenizer=self.tokenizer,
                data_collator=self.get_data_collator(),
            )

    def train(self):
        """Trains the model using LoRA fine-tuning."""
        print("Starting LoRA fine-tuning...")
        print(f"Using device: {self.device}")
        print(f"Task type: {self.task_type}")
        print(f"Training size: {len(self.tokenized_dataset['train'])} | Val size: {len(self.tokenized_dataset['val'])} | Test size: {len(self.tokenized_dataset['test'])}")

        self.trainer.train()

        cp_path = str(self.project_root / self.config['model']['cp_dir'])
        self.trainer.save_model(cp_path)
        self.tokenizer.save_pretrained(cp_path)
        print(f"Saving LoRA adapters to {cp_path}")
        self._save_training_logs()
        self._plot_losses()

    def test(self, size: int = 100, get_base_model: bool = True):
        """
        Evaluates the model on the test set.
        size: number of samples to evaluate.
        get_base_model: whether to also evaluate the untuned base model for comparison.
        """
        print(f"Using device: {self.device}")
        test_dataset = self.tokenized_dataset['test']
        actual_size = min(size, len(test_dataset))
        print(f"Testing model on {actual_size} samples...")
        test_dataset = test_dataset.select(range(actual_size))

        # 1. Evaluate LoRA model
        print("Evaluating LoRA model...")
        lora_outputs = self.trainer.predict(test_dataset)

        lora_predictions = np.where(lora_outputs.predictions != -100, lora_outputs.predictions, self.tokenizer.pad_token_id)
        lora_label_ids = np.where(lora_outputs.label_ids != -100, lora_outputs.label_ids, self.tokenizer.pad_token_id)

        lora_preds = self.tokenizer.batch_decode(lora_predictions, skip_special_tokens=True)
        lora_labels = self.tokenizer.batch_decode(lora_label_ids, skip_special_tokens=True)

        all_metrics = {"lora_metrics": lora_outputs.metrics}
        base_preds = None

        # 2. Evaluate base model (optional)
        if get_base_model:
            print("Evaluating Base model (no LoRA)...")
            original_model = self.trainer.model
            base_model = self._load_base_model().to(self.device)
            self.trainer.model = base_model

            base_outputs = self.trainer.predict(test_dataset)
            base_predictions = np.where(base_outputs.predictions != -100, base_outputs.predictions, self.tokenizer.pad_token_id)
            base_preds = self.tokenizer.batch_decode(base_predictions, skip_special_tokens=True)
            all_metrics["base_metrics"] = base_outputs.metrics

            self.trainer.model = original_model
            del base_model

        # 3. Sample comparison
        print("\n--- Sample Predictions Comparison ---")
        for i in range(min(2, len(lora_preds))):
            print(f"Target: {lora_labels[i]}")
            if base_preds:
                print(f"Base  : {base_preds[i]}")
            print(f"LoRA  : {lora_preds[i]}")
            print("-" * 30)

        # 4. Save results
        output_dir = self.project_root / self.config['dataset']['test_output_dir']
        os.makedirs(output_dir, exist_ok=True)
        model_name_safe = self.config['model']['model_name'].replace("/", "_")

        metrics_path = os.path.join(output_dir, f"{model_name_safe}_test_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(all_metrics, f, indent=4)
        print(f"Test metrics saved to {metrics_path}")

        lora_results = [{"label": lbl, "prediction": pred} for lbl, pred in zip(lora_labels, lora_preds)]
        lora_path = os.path.join(output_dir, f"{model_name_safe}_lora_predictions.json")
        with open(lora_path, "w") as f:
            json.dump(lora_results, f, indent=4)
        print(f"LoRA predictions saved to {lora_path}")

        if base_preds:
            base_results = [{"label": lbl, "prediction": pred} for lbl, pred in zip(lora_labels, base_preds)]
            base_path = os.path.join(output_dir, f"{model_name_safe}_base_predictions.json")
            with open(base_path, "w") as f:
                json.dump(base_results, f, indent=4)
            print(f"Base model predictions saved to {base_path}")

        return {
            "metrics": all_metrics,
            "lora_predictions": lora_preds,
            "base_predictions": base_preds,
            "labels": lora_labels,
        }

    def _save_training_logs(self):
        logs = self.trainer.state.log_history
        cp_path = str(self.project_root / self.config['model']['cp_dir'])
        logs_path = os.path.join(cp_path, "training_logs.json")
        with open(logs_path, "w") as f:
            json.dump(logs, f, indent=4)
        print(f"Training logs saved to {logs_path}")

    def _extract_metrics(self):
        logs = self.trainer.state.log_history
        train_losses, eval_losses, steps = [], [], []
        rogues = {"rouge1": [], "rouge2": [], "rougeL": [], "rougeLsum": []}

        for entry in logs:
            if "loss" in entry:
                train_losses.append(entry["loss"])
                steps.append(entry["step"])
            if "eval_loss" in entry:
                eval_losses.append(entry["eval_loss"])
            for key in rogues:
                if f"eval_{key}" in entry:
                    rogues[key].append(entry[f"eval_{key}"])

        return steps, train_losses, eval_losses, rogues

    def _plot_losses(self):
        steps, train_losses, eval_losses, rogues = self._extract_metrics()
        cp_path = str(self.project_root / self.config['model']['cp_dir'])

        plt.figure(figsize=(10, 6))
        plt.plot(steps, train_losses, label="Training Loss", color='blue', marker='o', markersize=5, linewidth=3)
        if eval_losses:
            plt.plot(steps, eval_losses, label="Validation Loss", color='orange', marker='o', markersize=5, linewidth=3)
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.savefig(os.path.join(cp_path, "loss_curve.png"))

        if any(rogues.values()):
            plt.figure(figsize=(10, 6))
            for rouge_name, scores in rogues.items():
                if scores:
                    plt.plot(steps, scores, label=rouge_name, marker='o', markersize=5, linewidth=3)
            plt.xlabel("Steps")
            plt.ylabel("Score")
            plt.title("ROUGE Metrics")
            plt.legend()
            plt.savefig(os.path.join(cp_path, "rouge_metrics.png"))

        plt.close()
        print(f"Training plots saved to {cp_path}")
