import numpy as np
import torch

from methods import AumTracker, DataMapTracker, ForgettingTracker, LossTracker
from transformers import Trainer

class CustomTrainer(Trainer):
    def __init__(self, *args, methods=None, num_classes=3, device=None, **kwargs):
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
        self.true_labels = [None] * self.total_samples

        # Epoch tracking
        self.current_epoch = 0
        self.device = device

    def get_train_dataloader(self):
        """
        Returns the training DataLoader, ensuring dataset indices are included in each batch.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_sampler = self._get_train_sampler() if not isinstance(self.train_dataset, torch.utils.data.IterableDataset) else None

        def collate_fn(batch):
            """Custom collate function to extract indices from dataset items."""
            # for item in batch:
            #     if isinstance(item['metadata'], list):
            #         item['metadata'] = torch.tensor(item['metadata'], dtype=torch.long)
            batch_dict = self.data_collator([{k: v for k, v in item.items() if k != "idx"} for item in batch])
            batch_indices = [item["idx"] for item in batch]
            batch_dict['idx'] = batch_indices
            # print(batch_dict)
            return batch_dict

        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=collate_fn,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


    def training_step(self, model, inputs, num_items_in_batch=None):
        """Track per-sample losses and predictions during training step"""
        # print(inputs)
        # Get outputs from parent class
        dataset_indices = inputs['idx']
        model_inputs = {key: value.to(self.device) for key, value in inputs.items() if key != 'idx'}
        loss = super().training_step(model, model_inputs, num_items_in_batch)

        if dataset_indices is not None:
            # print("here")
            dataset_indices = [idx.item() for idx in dataset_indices]
        # print("INDICES: ", dataset_indices)

        if self.methods:
            with torch.no_grad():
                outputs = model(**model_inputs)
                logits = outputs.logits
                labels = model_inputs["labels"]
                
                # Track predictions and forgetting
                predictions = torch.argmax(logits, dim=-1).cpu().numpy()
                labels_cpu = labels.cpu().numpy()

                # Update predictions
                while len(self.predictions) <= self.current_epoch:
                    self.predictions.append([None] * self.total_samples)
                for i, idx in enumerate(dataset_indices):
                    self.predictions[self.current_epoch][idx] = predictions[i]
                    if self.true_labels[idx] is None:
                        self.true_labels[idx] = labels_cpu[i]

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