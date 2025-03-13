import numpy as np
import torch

from methods import AumTracker, DataMapTracker, EL2NTracker, ForgettingTracker, GrandTracker, LossTracker
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
        self.el2n_tracker = EL2NTracker(self.total_samples) if "el2n" in self.methods else None
        self.forgetting_tracker = ForgettingTracker(self.total_samples) if "forgetting" in self.methods else None
        self.grand_tracker = GrandTracker(self.total_samples) if "grand" in self.methods else None
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
        dataset_indices = inputs['idx']
        model_inputs = {key: value.to(self.device) for key, value in inputs.items() if key != 'idx'}
        model_inputs['output_hidden_states'] = True
        loss = super().training_step(model, model_inputs, num_items_in_batch)

        if dataset_indices is not None:
            dataset_indices = [idx.item() for idx in dataset_indices]

        if self.methods:
            with torch.enable_grad():  # Enable gradients from the start
                outputs = model(**model_inputs)
                logits = outputs.logits
                labels = model_inputs["labels"]

                predictions = torch.argmax(logits, dim=-1).cpu().numpy()
                labels_cpu = labels.cpu().numpy()

                # Update predictions
                while len(self.predictions) <= self.current_epoch:
                    self.predictions.append([None] * self.total_samples)
                for i, idx in enumerate(dataset_indices):
                    self.predictions[self.current_epoch][idx] = predictions[i]
                    if self.true_labels[idx] is None:
                        self.true_labels[idx] = labels_cpu[i]

                # Detach outputs for trackers that don't need gradients
                detached_logits = logits.detach()
                detached_probs = torch.nn.functional.softmax(detached_logits, dim=-1)

                # Update trackers
                if self.aum_tracker:
                    self.aum_tracker.update(detached_logits, labels, dataset_indices)
                if self.data_map_tracker:
                    self.data_map_tracker.update(dataset_indices, detached_logits, labels, detached_probs)
                if self.el2n_tracker:
                    self.el2n_tracker.update(dataset_indices, detached_probs, labels)
                if self.forgetting_tracker:
                    correct_predictions = (predictions == labels_cpu)
                    self.forgetting_tracker.update(correct_predictions, dataset_indices)
                if self.loss_tracker:
                    self.loss_tracker.update(logits=detached_logits, labels=labels, dataset_indices=dataset_indices)

                # Update GraNd tracker
                if self.grand_tracker:
                    self.grand_tracker.update(dataset_indices, logits, labels, self.model) 

        return loss

    def evaluate(self, *args, **kwargs):
        """On end of epoch, finalise epoch loss tracking."""
        if self.loss_tracker:
            self.loss_tracker.finalise_epoch()
        if self.data_map_tracker:
            self.data_map_tracker.finalise_epoch()
        if self.aum_tracker:
            self.aum_tracker.finalise_epoch()
        if self.el2n_tracker:
            self.el2n_tracker.finalise_epoch()
        if self.grand_tracker:
            self.grand_tracker.finalise_epoch()
        self.current_epoch += 1
        return super().evaluate(*args, **kwargs)

    def get_unified_stats(self):
        """Return stats for enabled tracking methods"""
        stats = {}
        if self.aum_tracker:
            stats['aum_stats'] = self.aum_tracker.get_stats()
        if self.data_map_tracker:
            stats['data_map_stats'] = self.data_map_tracker.get_stats()
        if self.el2n_tracker:
            stats["el2n_scores"] = self.el2n_tracker.get_scores()
        if self.forgetting_tracker:
            stats['forgetting_stats'] = self.forgetting_tracker.get_stats()
        if self.grand_tracker:
            stats["grand_scores"] = self.grand_tracker.get_scores()
        if self.loss_tracker:
            stats['loss_stats'] = self.loss_tracker.get_stats()

        stats['predictions'] = self.predictions
        stats['true_labels'] = self.true_labels
        return stats