from typing import Union, Dict, Optional
from .FlanT5Base import FlanT5Base
from .summarization_dataset import SummarizationDataset
from utils.utils import filter_training_args

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import numpy as np
import matplotlib.pyplot as plt
import json
import os

class FlanT5Trainer(FlanT5Base):
    """
    Class for fine-tuning, testing and inference of FlanT5 model.
    """
    def __init__(self, config: Union[str, Dict] = "flan-t5"):
        """
        config: str or dict, name of the config, path to YAML, or the config dict themselves. Default to "flan-t5".
        """
        super().__init__(config)
        self.tokenized_dataset = None
        self.trainer = None
        self.args = filter_training_args(self.config['seq2seq_args'])
        self.model.print_trainable_parameters()

    def prepare_data_and_trainer(self, training_args_dict: Optional[Dict] = None, **kwargs):
        """
        Loads and tokenizes the dataset using this trainer's own tokenizer.
        training_args_dict: dict, optional training arguments to override/extend config.
        kwargs can be used to override train_size, val_size, test_size, etc.
        """
        print("Preparing dataset...")
        
        ds_manager = SummarizationDataset(
            tokenizer=self.tokenizer,
            config=self.config,
            **kwargs
        )
        self.tokenized_dataset = ds_manager.load_and_prepare()
        self.trainer = self._get_trainer(training_args_dict)

    def _get_trainer(self, training_args_dict: Optional[Dict] = None):
        """
        Returns a Seq2SeqTrainer instance.
        """
        if self.tokenized_dataset is None:
            raise ValueError("Dataset not prepared. Call prepare_data() first.")
        
        # Use config args by default, merge with custom dict if provided
        final_args = self.args.copy()
        if training_args_dict:
            print("Merging custom training arguments...")
            custom_args = filter_training_args(training_args_dict)
            final_args.update(custom_args)
            
        training_args = Seq2SeqTrainingArguments(
            **final_args
        )

        return Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_dataset['train'],
            eval_dataset=self.tokenized_dataset['val'],
            tokenizer=self.tokenizer,
            data_collator=self.get_data_collator(),
            compute_metrics=self.compute_metrics
        )

    def train(self):
        """
        Trains the model using the prepared dataset with LoRA.
        """
        print("Starting LoRA fine-tuning...")
        print(f"Using device: {self.device}")
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
        Tests the model on the test set.
        size: int, number of samples to test on. Default to 100.
        get_base_model: bool, whether to evaluate the base model. Default to True.
        """
        print(f"Using device: {self.device}")
        test_dataset = self.tokenized_dataset['test']
        actual_size = min(size, len(test_dataset))
        print(f"Testing model on {actual_size} samples...")
        test_dataset = test_dataset.select(range(actual_size))

        # 1. Evaluate LoRA Model
        print("Evaluating LoRA model...")
        lora_outputs = self.trainer.predict(test_dataset)
        
        # Replace -100 padding with pad_token_id before decoding
        lora_predictions = np.where(lora_outputs.predictions != -100, lora_outputs.predictions, self.tokenizer.pad_token_id)
        lora_label_ids = np.where(lora_outputs.label_ids != -100, lora_outputs.label_ids, self.tokenizer.pad_token_id)

        lora_preds = self.tokenizer.batch_decode(lora_predictions, skip_special_tokens=True)
        lora_labels = self.tokenizer.batch_decode(lora_label_ids, skip_special_tokens=True)
        
        all_metrics = {"lora_metrics": lora_outputs.metrics}
        base_preds = None

        # 2. Evaluate Base Model (if requested)
        if get_base_model:
            print("Evaluating Base model (no LoRA)...")
            # Temporarily swap model in trainer
            original_model = self.trainer.model
            base_model = self._load_base_model().to(self.device)
            self.trainer.model = base_model
            
            base_outputs = self.trainer.predict(test_dataset)
            
            # Replace -100 padding with pad_token_id before decoding
            base_predictions = np.where(base_outputs.predictions != -100, base_outputs.predictions, self.tokenizer.pad_token_id)
            base_preds = self.tokenizer.batch_decode(base_predictions, skip_special_tokens=True)
            
            all_metrics["base_metrics"] = base_outputs.metrics
            
            # Revert trainer model
            self.trainer.model = original_model
            del base_model # Free memory

        # 3. Print Sample Comparison
        print("\n--- Sample Predictions Comparison ---")
        for i in range(min(2, len(lora_preds))):
            print(f"Target: {lora_labels[i]}")
            if base_preds:
                print(f"Base  : {base_preds[i]}")
            print(f"LoRA  : {lora_preds[i]}")
            print("-" * 30)

        # 4. Save Results
        output_dir = self.project_root / self.config['dataset']['test_output_dir']
        os.makedirs(output_dir, exist_ok=True)
        model_name_safe = self.config['model']['model_name'].replace("/", "_")
        
        # Save Metrics
        metrics_path = os.path.join(output_dir, f"{model_name_safe}_test_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(all_metrics, f, indent=4)
        print(f"Test metrics saved to {metrics_path}")

        # Save LoRA Predictions
        lora_results = [
            {"label": label, "prediction": preds} 
            for label, preds in zip(lora_labels, lora_preds)
        ]
        lora_path = os.path.join(output_dir, f"{model_name_safe}_lora_predictions.json")
        with open(lora_path, "w") as f:
            json.dump(lora_results, f, indent=4)
        print(f"LoRA predictions saved to {lora_path}")
        
        # Save Base Predictions
        if base_preds:
            base_results = [
                {"label": label, "prediction": preds} 
                for label, preds in zip(lora_labels, base_preds)
            ]
            base_path = os.path.join(output_dir, f"{model_name_safe}_base_predictions.json")
            with open(base_path, "w") as f:
                json.dump(base_results, f, indent=4)
            print(f"Base model predictions saved to {base_path}")

        return {
            "metrics": all_metrics,
            "lora_predictions": lora_preds,
            "base_predictions": base_preds,
            "labels": lora_labels
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
        train_losses = []
        eval_losses = []
        rogues = {
            "rouge1": [],
            "rouge2": [],
            "rougeL": [],
            "rougeLsum": []
        }
        steps = []

        for entry in logs:
            if "loss" in entry:
                train_losses.append(entry["loss"])
                steps.append(entry["step"])
            if "eval_loss" in entry:
                eval_losses.append(entry["eval_loss"])
            if "eval_rouge1" in entry:
                rogues["rouge1"].append(entry["eval_rouge1"])
            if "eval_rouge2" in entry:
                rogues["rouge2"].append(entry["eval_rouge2"])
            if "eval_rougeL" in entry:
                rogues["rougeL"].append(entry["eval_rougeL"])
            if "eval_rougeLsum" in entry:
                rogues["rougeLsum"].append(entry["eval_rougeLsum"])
        
        return steps, train_losses, eval_losses, rogues
                
    def _plot_losses(self):
        steps, train_losses, eval_losses, rogues = self._extract_metrics()

        cp_path = str(self.project_root / self.config['model']['cp_dir'])

        plt.figure(figsize=(10, 6))
        plt.plot(steps, train_losses, label="Training Loss", color='blue', marker='o', markersize=5, linewidth=3)
        if eval_losses:
            plt.plot(steps, eval_losses, label="Validation loss", color='orange', marker='o', markersize=5, linewidth=3)
        
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        loss_plot_path = os.path.join(cp_path, "loss_curve.png")
        plt.savefig(loss_plot_path)

        if rogues:
            plt.figure(figsize=(10, 6))
            for rouge_name, scores in rogues.items():
                if scores:
                    plt.plot(steps, scores, label=rouge_name, marker='o', markersize=5, linewidth=3)
            plt.xlabel("Steps")
            plt.ylabel("Score")
            plt.title("ROUGE Metrics")
            plt.legend()
            rouge_plot_path = os.path.join(cp_path, "rouge_metrics.png")
            plt.savefig(rouge_plot_path)

        plt.close()
        print(f"Training plots saved to {cp_path}")


    
