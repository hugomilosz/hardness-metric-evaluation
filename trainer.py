import time
from datetime import timedelta
from typing import List, Dict, Optional, Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from accelerate import Accelerator
from evaluate import load as load_metric
import wandb

from methods import (
    AumTracker,
    DataMapTracker,
    EL2NTracker,
    ForgettingTracker,
    GrandTracker,
    LossTracker,
)

class Trainer:
    """
    Handles the training and evaluation of a Transformer model.

    This class manages the entire training pipeline, from data loading and
    model optimisation to evaluation. It uses Hugging Face's Accelerate for
    seamless multi-GPU and mixed-precision training. It can also track various
    data-centric metrics throughout training.
    """

    def __init__(
        self,
        dataset_bundle: Any,
        args: Any,
        methods: Optional[List[str]] = None,
        eval_metrics: Optional[List[str]] = None,
    ):
        """
        Initialises the Trainer.

        Args:
            dataset_bundle (Any): An object with attributes like model, train_dataset,
                                  eval_dataset, model_name, and num_labels.
            args (Any): A configuration object (e.g., argparse.Namespace) with
                        hyperparameters like learning_rate, batch_size, etc.
            methods (Optional[List[str]]): A list of data tracking methods to use,
                                           e.g., ["aum", "datamaps"].
            eval_metrics (Optional[List[str]]): Metrics to compute during evaluation,
                                                e.g., ["accuracy", "f1"].
        """
        self.accelerator = Accelerator(mixed_precision=args.mixed_precision)
        self.args = args
        self.seed = args.seed

        self.model = dataset_bundle.model
        self.train_dataset = dataset_bundle.train_dataset
        self.eval_dataset = dataset_bundle.eval_dataset
        self.model_name = dataset_bundle.model_name
        self.num_classes = dataset_bundle.num_labels
        self.total_samples = len(self.train_dataset)
        self.dataset_name = dataset_bundle.dataset_name

        self.methods = methods or []
        self.eval_metrics = eval_metrics or []
        self.num_epochs = self.args.num_train_epochs
        self.current_epoch = 0
        self.device = self.accelerator.device

        self._initialise_trackers()

        # Pre-allocate arrays for storing results across epochs.
        self.true_labels = np.full(self.total_samples, np.nan, dtype=np.int32)
        self.predictions = np.full((self.num_epochs, self.total_samples), np.nan, dtype=np.int32)

        # Load metric computation objects from the `evaluate` library.
        self.metric_objects = {name: load_metric(name) for name in self.eval_metrics}

    def _initialise_trackers(self):
        """Sets up the requested data-centric AI trackers based on the methods list."""
        if "aum" in self.methods:
            self.aum_tracker = AumTracker(self.total_samples, self.num_classes)
        if "datamaps" in self.methods:
            self.data_map_tracker = DataMapTracker(self.total_samples)
        if "el2n" in self.methods:
            self.el2n_tracker = EL2NTracker(self.total_samples)
        if "forgetting" in self.methods:
            self.forgetting_tracker = ForgettingTracker(self.total_samples)
        if "loss" in self.methods:
            self.loss_tracker = LossTracker(self.total_samples)
        if "grand" in self.methods:
            # GraNd requires access to the model's classifier layer.
            classifier_module = getattr(self.model, "logits_proj", self.model.classifier)
            self.grand_tracker = GrandTracker(self.model, classifier_module, self.total_samples)

    def get_dataloader(self, dataset: torch.utils.data.Dataset, batch_size: int, shuffle: bool = True) -> DataLoader:
        """Creates a DataLoader with a fixed seed for reproducibility."""
        generator = torch.Generator().manual_seed(self.seed)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=True,
            generator=generator,
        )

    def prepare_training(self) -> (DataLoader, DataLoader):
        """
        Prepares the model, optimiser, schedulers, and dataloaders for training.

        This method leverages `accelerator.prepare()` to wrap all components,
        making them ready for distributed environments.
        """
        train_dataloader = self.get_dataloader(self.train_dataset, self.args.train_batch_size)
        eval_dataloader = self.get_dataloader(self.eval_dataset, self.args.eval_batch_size, shuffle=False)

        # Exclude bias and LayerNorm weights from weight decay.
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon,
        )

        total_steps = (len(train_dataloader) // self.args.gradient_accumulation_steps) * self.num_epochs
        warmup_steps = int(self.args.warmup_ratio * total_steps) if isinstance(self.args.warmup_ratio, float) else self.args.warmup_ratio

        self.scheduler = get_scheduler(
            name=self.args.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # Use accelerator to prepare all components.
        self.model, self.optimizer, train_dataloader, eval_dataloader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, train_dataloader, eval_dataloader, self.scheduler
        )

        # Optional: compile the model for a speed-up on newer hardware.
        if hasattr(torch, 'compile') and torch.cuda.get_device_capability()[0] >= 7:
            self.model = torch.compile(self.model)

        return train_dataloader, eval_dataloader

    def train(self):
        """The main training loop."""
        train_dataloader, eval_dataloader = self.prepare_training()

        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0.0
            start_time = time.time()
            num_batches = len(train_dataloader)

            for step, batch in enumerate(train_dataloader):
                with self.accelerator.accumulate(self.model):
                    dataset_indices = batch.pop("idx")
                    outputs = self.model(**batch, output_hidden_states=True)
                    loss = torch.nn.functional.cross_entropy(
                        outputs.logits.view(-1, self.num_classes),
                        batch["labels"].view(-1),
                        reduction="mean"
                    )

                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                total_loss += loss.item() * self.args.gradient_accumulation_steps
                self.track_metrics(outputs, batch, dataset_indices)

                if (step + 1) % 50 == 0 or (step + 1) == num_batches:
                    self._log_progress(epoch, step + 1, num_batches, start_time)
            
            avg_loss = total_loss / len(train_dataloader)
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}, Average Training Loss: {avg_loss:.4f}")

            self.finalise_epoch()
            eval_results = self.evaluate(eval_dataloader)
            print(f"Epoch {epoch + 1} Evaluation: {eval_results}")

    def track_metrics(self, model_outputs: Dict, batch: Dict, dataset_indices: torch.Tensor):
        """Updates all active trackers with the results from a single step."""
        logits = model_outputs.logits.detach()
        labels = batch["labels"]
        
        # Gather results from all processes in a distributed setting.
        all_logits = self.accelerator.gather(logits)
        all_labels = self.accelerator.gather(labels)
        all_indices = self.accelerator.gather(dataset_indices)
        
        predictions = torch.argmax(all_logits, dim=-1).cpu().numpy()
        probs = torch.nn.functional.softmax(all_logits, dim=-1).cpu().numpy()
        np_labels = all_labels.cpu().numpy()
        np_indices = all_indices.cpu().numpy()

        # Store predictions and true labels.
        self.predictions[self.current_epoch, np_indices] = predictions
        unseen_mask = np.isnan(self.true_labels[np_indices])
        self.true_labels[np_indices[unseen_mask]] = np_labels[unseen_mask]

        if "aum" in self.methods:
            self.aum_tracker.update(all_logits, all_labels, all_indices)
        if "datamaps" in self.methods:
            self.data_map_tracker.update(all_indices, all_logits, all_labels, probs)
        if "el2n" in self.methods:
            self.el2n_tracker.update(all_indices, probs, all_labels)
        if "forgetting" in self.methods:
            correct = (predictions == np_labels)
            self.forgetting_tracker.update(correct, np_indices)
        if "loss" in self.methods:
            self.loss_tracker.update(logits=all_logits, labels=all_labels, dataset_indices=all_indices)
        if "grand" in self.methods:
            # Extract the correct hidden state based on the model type.
            if "roberta" in self.model_name or "bert" in self.model_name:
                features = model_outputs.hidden_states[-1][:, 0, :]
            elif "xlnet" in self.model_name:
                features = model_outputs.hidden_states[-1][:, -1, :]
            
            self.grand_tracker.update(
                model_inputs={"features": features, "labels": labels},
                dataset_indices=dataset_indices,
                device=self.device
            )

    def finalise_epoch(self):
        """Calls the end-of-epoch finalisation for each active tracker."""
        tracker_list = [
            getattr(self, name, None) for name in
            ["loss_tracker", "data_map_tracker", "aum_tracker", "el2n_tracker", "grand_tracker", "forgetting_tracker"]
        ]
        for tracker in tracker_list:
            if tracker:
                tracker.finalise_epoch()
        self.current_epoch += 1

    def evaluate(self, eval_dataloader: DataLoader) -> Dict[str, float]:
        """Evaluates the model on a given dataloader."""
        self.model.eval()
        all_preds, all_labels = [], []
        total_eval_loss = 0.0

        with torch.no_grad():
            for batch in eval_dataloader:
                labels = batch.pop("labels")
                outputs = self.model(**batch)
                loss = torch.nn.functional.cross_entropy(
                    outputs.logits.view(-1, self.num_classes),
                    labels.view(-1)
                )
                
                total_eval_loss += self.accelerator.gather(loss).mean().item()
                preds = torch.argmax(outputs.logits, dim=-1)
                
                all_preds.extend(self.accelerator.gather(preds).cpu().numpy())
                all_labels.extend(self.accelerator.gather(labels).cpu().numpy())
        
        metrics = {"loss": total_eval_loss / len(eval_dataloader)}
        for name, metric in self.metric_objects.items():
            # Some metrics like F1 require a specific averaging strategy.
            if name in ["f1", "precision", "recall"]:
                result = metric.compute(predictions=all_preds, references=all_labels, average="macro")
            else:
                result = metric.compute(predictions=all_preds, references=all_labels)
            
            if isinstance(result, dict):
                metrics.update(result)
            else:
                metrics[name] = result

        # Log metrics to W&B on the main process.
        if self.accelerator.is_main_process:
            wandb.log({f"eval/{k}": v for k, v in metrics.items()}, step=self.current_epoch)
            
        return metrics

    def get_unified_stats(self) -> Dict[str, np.ndarray]:
        """Consolidates and returns all tracked statistics after training."""
        stats = {}
        if "aum" in self.methods:
            stats['aum'] = self.aum_tracker.get_stats()
        if "datamaps" in self.methods:
            stats['datamap'] = self.data_map_tracker.get_stats()
        if "el2n" in self.methods:
            stats['el2n'] = self.el2n_tracker.get_stats()
        if "forgetting" in self.methods:
            stats['forgetting'] = self.forgetting_tracker.get_stats()
        if "grand" in self.methods:
            stats['grand'] = self.grand_tracker.get_stats()
        if "loss" in self.methods:
            stats['loss'] = self.loss_tracker.get_stats()

        stats['predictions'] = self.predictions
        stats['true_labels'] = self.true_labels
        return stats

    def _log_progress(self, epoch: int, step: int, total_steps: int, start_time: float):
        """Prints a progress bar to the console."""
        elapsed = time.time() - start_time
        eta_seconds = (elapsed / step) * (total_steps - step)
        eta_str = str(timedelta(seconds=int(eta_seconds)))
        percentage = (step / total_steps) * 100
        
        print(
            f"Epoch {epoch+1}/{self.num_epochs} | Step {step}/{total_steps} | "
            f"{percentage:.2f}% | Elapsed: {timedelta(seconds=int(elapsed))} | "
            f"ETA: {eta_str}", end='\r'
        )