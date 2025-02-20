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
        
        # Epoch tracking
        self.global_step = 0
        self.current_epoch = 0

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs["labels"]
        
        per_sample_loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        per_sample_losses = per_sample_loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        batch_size = labels.size(0)
        start_idx = (self.global_step % (self.total_samples // self.args.per_device_train_batch_size)) * self.args.per_device_train_batch_size
        dataset_indices = [(start_idx + i) % self.total_samples for i in range(batch_size)]
        
        predictions = torch.argmax(logits, dim=-1)
        correct_predictions = (predictions == labels).cpu().numpy()

        # Update trackers if enabled
        if self.loss_tracker:
            self.loss_tracker.update_batch(
                [loss.item() for loss in per_sample_losses],
                dataset_indices
            )
            
        if self.forgetting_tracker:
            self.forgetting_tracker.update(correct_predictions, dataset_indices)

        self.global_step += 1
        # mean_loss = per_sample_losses.mean()
        # return (mean_loss, outputs) if return_outputs else mean_loss

        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        if return_outputs:
            return loss, outputs
        return loss

    def training_step(self, *args, **kwargs):
        """Track per-batch loss."""
        loss = super().training_step(*args, **kwargs)
        if self.loss_tracker:
            self.loss_tracker.update_epoch_loss(loss.detach().cpu().item())
        return loss

    def evaluate(self, *args, **kwargs):
        """On end of epoch, finalise epoch loss tracking."""
        if self.loss_tracker:
            epoch_avg_loss = self.loss_tracker.finalise_epoch()
            if epoch_avg_loss is not None:
                print(f"\nEpoch {len(self.loss_tracker.epoch_losses)} Summary:")
                print(f"Average Loss: {epoch_avg_loss:.4f}")

        return super().evaluate(*args, **kwargs)

    def get_unified_stats(self):
        """Return stats for enabled tracking methods"""
        stats = {}
        if self.loss_tracker:
            stats['loss_stats'] = self.loss_tracker.get_stats()
        if self.forgetting_tracker:
            stats['forgetting_stats'] = self.forgetting_tracker.get_stats()
            
        return stats