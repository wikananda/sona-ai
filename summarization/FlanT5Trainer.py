from FlanT5Base import FlanT5Base
from utils.utils import filter_training_args
from transformers import Seq2SeqTrainer
import json
import os

class FlanT5Trainer(FlanT5Base):
    def __init__(self, tokenized_dataset):
        super().__init__("flan-t5")
        self.tokenized_dataset = tokenized_dataset
        self.trainer = self._get_trainer()
        self.args = filter_training_args(self.config['seq2seq_args'])

    def _get_trainer(self):
        return Seq2SeqTrainer(
            model=self.model,
            args=self.args,
            train_dataset=self.tokenized_dataset['train'],
            eval_dataset=self.tokenized_dataset['val'],
            tokenizer=self.tokenizer,
            data_collator=self.get_data_collator(),
            compute_metrics=self.compute_metrics
        )

    def train(self):
        print("Starting LoRA fine-tuning...")
        self.trainer.train()
        self.trainer.save_model(self.config['model']['cp_dir'])
        self.tokenizer.save_pretrained(self.config['model']['cp_dir'])
        print(f"Saving LoRA adapters to {self.config['model']['cp_dir']}")

    def test(self, size: int = 100, get_base_model: bool = True):
        print(f"Testing model on {size} samples...")
        test_dataset = self.tokenized_dataset['test']
        if size < len(test_dataset):
            test_dataset = test_dataset.select(range(size))

        # 1. Evaluate LoRA Model
        print("Evaluating LoRA model...")
        lora_outputs = self.trainer.predict(test_dataset)
        lora_preds = self.tokenizer.batch_decode(lora_outputs.predictions, skip_special_tokens=True)
        lora_labels = self.tokenizer.batch_decode(lora_outputs.label_ids, skip_special_tokens=True)
        
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
            base_preds = self.tokenizer.batch_decode(base_outputs.predictions, skip_special_tokens=True)
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
        output_dir = self.config['dataset']['test_output_dir']
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
