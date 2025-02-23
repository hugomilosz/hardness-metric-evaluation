import torch
from transformers import Trainer
import numpy as np
from methods import LossTracker
from methods import ForgettingTracker

class CustomTrainer(Trainer):
    def __init__(self, *args, methods=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.methods = methods or []
        self.total_samples = len(self.train_dataset)

        # Initialise trackers based on methods
        self.loss_tracker = LossTracker(self.total_samples) if "loss" in self.methods else None
        self.forgetting_tracker = ForgettingTracker(self.total_samples) if "forgetting" in self.methods else None
        self.predictions = []
        self.true_labels = []

        # Epoch tracking
        self.global_step = 0
        self.current_epoch = 0

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Track per-sample losses and predictions during training step"""
        # Get outputs from parent class
        loss = super().training_step(model, inputs, num_items_in_batch)
        
        if self.methods:
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                labels = inputs["labels"]

                batch_size = labels.size(0)
                start_idx = (self.global_step % (self.total_samples // self.args.per_device_train_batch_size)) * self.args.per_device_train_batch_size
                dataset_indices = [(start_idx + i) % self.total_samples for i in range(batch_size)]
                
                # Track predictions and forgetting
                predictions = torch.argmax(logits, dim=-1)
                self.predictions.extend(predictions.cpu().numpy())
                self.true_labels.extend(labels.cpu().numpy())
                
                # Update loss tracker
                if self.loss_tracker:
                    batch_loss = self.loss_tracker.update_from_batch(
                        logits=logits,
                        labels=labels,
                        dataset_indices=dataset_indices,
                    )
                    # self.loss_tracker.update_epoch_loss(batch_loss)

                # Update forgetting tracker
                if self.forgetting_tracker:
                    correct_predictions = (predictions == labels).cpu().numpy()
                    self.forgetting_tracker.update(correct_predictions, dataset_indices)

        self.global_step += 1
        return loss

    def evaluate(self, *args, **kwargs):
        """On end of epoch, finalise epoch loss tracking."""
        if self.loss_tracker:
            epoch_avg_loss = self.loss_tracker.finalise_epoch()
        return super().evaluate(*args, **kwargs)

    def get_unified_stats(self):
        """Return stats for enabled tracking methods"""
        stats = {}
        if self.loss_tracker:
            stats['loss_stats'] = self.loss_tracker.get_stats()
        if self.forgetting_tracker:
            stats['forgetting_stats'] = self.forgetting_tracker.get_stats()
        stats['predictions'] = self.predictions
        return stats