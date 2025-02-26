import torch
from transformers import Trainer
import numpy as np
from methods import AumTracker, DataMapTracker, ForgettingTracker, LossTracker

class CustomTrainer(Trainer):
    def __init__(self, *args, methods=None, num_classes=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.methods = methods or []
        self.num_classes = num_classes
        self.total_samples = len(self.train_dataset)

        # Initialise trackers based on methods
        self.aum_tracker = AumTracker(self.num_classes) if "aum" in self.methods else None
        self.data_map_tracker = DataMapTracker(self.total_samples) if "datamaps" in self.methods else None
        self.forgetting_tracker = ForgettingTracker(self.total_samples) if "forgetting" in self.methods else None
        self.loss_tracker = LossTracker(self.total_samples) if "loss" in self.methods else None

        self.predictions = []
        self.true_labels = []

        # Epoch tracking
        self.current_epoch = 0
        self.global_step = 0

    def training_step(self, model, inputs, num_items_in_batch=None):
        # print("STEP")
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
                predictions = torch.argmax(logits, dim=-1).cpu().numpy()
                labels_cpu = labels.cpu().numpy()

                # Ensure predictions list has a sublist for current epoch
                while len(self.predictions) <= self.current_epoch:
                    self.predictions.append([])
                    self.true_labels.append([])

                # Store batch predictions for the current epoch
                self.predictions[self.current_epoch].extend(predictions)
                self.true_labels[self.current_epoch].extend(labels_cpu)

                # Update AUM tracker
                if self.aum_tracker:
                    self.aum_tracker.update(logits, labels, dataset_indices)
                
                # Update loss tracker
                if self.loss_tracker:
                    batch_loss = self.loss_tracker.update(
                        logits=logits,
                        labels=labels,
                        dataset_indices=dataset_indices,
                    )

                # Update forgetting tracker
                if self.forgetting_tracker:
                    correct_predictions = (predictions == labels.cpu())
                    self.forgetting_tracker.update(correct_predictions, dataset_indices)

                # Update data map tracker
                if self.data_map_tracker:
                    probabilities = torch.nn.functional.softmax(logits, dim=-1)
                    self.data_map_tracker.update(dataset_indices, logits, labels, probabilities)

        self.global_step += 1
        return loss

    def evaluate(self, *args, **kwargs):
        """On end of epoch, finalise epoch loss tracking."""
        if self.loss_tracker:
            epoch_avg_loss = self.loss_tracker.finalise_epoch()
            print(epoch_avg_loss)
        if self.data_map_tracker:
            self.data_map_tracker.finalise_epoch()
        if self.aum_tracker:
            self.aum_tracker.finalise_epoch()
        self.current_epoch += 1
        return super().evaluate(*args, **kwargs)

    def get_unified_stats(self):
        """Return stats for enabled tracking methods"""
        stats = {}
        if self.aum_tracker:
            stats['aum_stats'] = self.aum_tracker.get_stats()
        if self.data_map_tracker:
            stats['data_map_stats'] = self.data_map_tracker.get_stats()
        if self.forgetting_tracker:
            stats['forgetting_stats'] = self.forgetting_tracker.get_stats()
        if self.loss_tracker:
            stats['loss_stats'] = self.loss_tracker.get_stats()

        stats['predictions'] = self.predictions
        stats['true_labels'] = self.true_labels
        return stats