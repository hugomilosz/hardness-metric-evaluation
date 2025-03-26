<<<<<<< HEAD
import torch
from transformers import Trainer, get_linear_schedule_with_warmup
=======
>>>>>>> origin
import numpy as np
import torch

from methods import AumTracker, DataMapTracker, ForgettingTracker, LossTracker
from transformers import Trainer

<<<<<<< HEAD
import torch
from tqdm import tqdm
import time

class CustomTrainer:
    def __init__(self, model, train_dataset, eval_dataset=None, optimizer=None, loss_fn=None, methods=None, batch_size=32, shuffle=True, device=None):
        """
        Initialises the custom trainer.

        Args:
            model: The model to be trained.
            train_dataset: The training dataset.
            eval_dataset: The evaluation dataset (optional).
            optimizer: The optimizer used during training.
            loss_fn: The loss function used during training.
            methods: List of tracking methods like "loss" and "forgetting".
            batch_size: The batch size for training.
            shuffle: Whether to shuffle the dataset.
            device: The device to train the model on (e.g., 'cpu' or 'cuda').
        """
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.methods = methods or []
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.loss_tracker = LossTracker(len(train_dataset)) if "loss" in self.methods else None
        self.forgetting_tracker = ForgettingTracker(len(train_dataset)) if "forgetting" in self.methods else None

        # Move model to the correct device
        self.model.to(self.device)

    def _get_dataloader(self, dataset):
        """Helper function to get DataLoader with shuffling enabled."""
        def custom_collate_fn(batch):
            # Manually handle padding if needed
            input_ids = [item['input_ids'] for item in batch]
            attention_mask = [item['attention_mask'] for item in batch]
            labels = [item['labels'] for item in batch]
            idx = [item['idx'] for item in batch]
            
            # Pad sequences manually if needed
            input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
            attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
            labels = torch.tensor(labels)
            idx = torch.tensor(idx)

            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'idx': idx
            }
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle, collate_fn=custom_collate_fn)

    def _train_one_epoch(self):
        """Train the model for one epoch."""
        self.model.train()
        train_dataloader = self._get_dataloader(self.train_dataset)
        total_loss = 0
        total_correct = 0
        total_samples = 0

        scaler = torch.cuda.amp.GradScaler()
        start_time = time.time()

        for batch in tqdm(train_dataloader, desc="Training", ncols=100):
            self.optimizer.zero_grad()

            dataset_indices = batch.get('idx', None)
            if dataset_indices is not None:
                dataset_indices = dataset_indices.cpu().numpy().tolist()

            inputs = {key: value.to(self.device) for key, value in batch.items() if key != 'idx'}

            # Forward pass
            with torch.cuda.amp.autocast():
                outputs = self.model(**inputs)
                logits = outputs.logits
                labels = inputs['labels']
                loss = self.loss_fn(logits, labels)

            # Backward pass
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            self.scheduler.step()
            scaler.update()

            # Track loss and forgetting metrics
            total_loss += loss.item()

            # Track predictions and forgetting
            predictions = torch.argmax(logits, dim=-1)
            correct_predictions = (predictions == labels).sum().item()

            total_correct += correct_predictions
            total_samples += len(labels)

            # If needed, update the trackers
            if self.loss_tracker:
                # Ensure that the 'idx' field is passed to the tracker for loss calculation
                dataset_indices = batch.get('idx', None)
                if dataset_indices is not None:
                    self.loss_tracker.update_from_batch(logits, labels, dataset_indices.cpu().numpy().tolist())

            if self.forgetting_tracker:
                # Update forgetting tracker (correct predictions vs true labels)
                correct_predictions_np = (predictions == labels).cpu().numpy()
                dataset_indices = batch.get('idx', None)
                if dataset_indices is not None:
                    self.forgetting_tracker.update(correct_predictions_np, dataset_indices.cpu().numpy().tolist())
=======
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
>>>>>>> origin

        if self.loss_tracker:
<<<<<<< HEAD
            self.loss_tracker.finalise_epoch()
=======
            epoch_avg_loss = self.loss_tracker.finalise_epoch()
            print(epoch_avg_loss)
        if self.data_map_tracker:
            self.data_map_tracker.finalise_epoch()
        if self.aum_tracker:
            self.aum_tracker.finalise_epoch()
        self.current_epoch += 1
        return super().evaluate(*args, **kwargs)
>>>>>>> origin

        avg_loss = total_loss / len(train_dataloader)
        accuracy = total_correct / total_samples
        return avg_loss, accuracy

    def _evaluate(self):
        """Evaluate the model."""
        self.model.eval()
        eval_dataloader = self._get_dataloader(self.eval_dataset) if self.eval_dataset else None
        total_loss = 0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating", ncols=100):
                inputs = {key: value.to(self.device) for key, value in batch.items() if key != 'idx'}

                # Forward pass
                outputs = self.model(**inputs)
                logits = outputs.logits
                labels = inputs['labels']

                # Calculate loss
                loss = self.loss_fn(logits, labels)
                total_loss += loss.item()

                # Track predictions
                predictions = torch.argmax(logits, dim=-1)
                correct_predictions = (predictions == labels).sum().item()

                total_correct += correct_predictions
                total_samples += len(labels)

        avg_loss = total_loss / len(eval_dataloader)
        accuracy = total_correct / total_samples
        return avg_loss, accuracy

    def train(self, epochs):
        """Train the model for the specified number of epochs."""
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, 
            num_warmup_steps=0,
            num_training_steps=len(self.train_dataset) * 3 // self.batch_size
        )

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            train_loss, train_accuracy = self._train_one_epoch()
            print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}")

            if self.eval_dataset:
                eval_loss, eval_accuracy = self._evaluate()
                print(f"Evaluation Loss: {eval_loss:.4f}, Evaluation Accuracy: {eval_accuracy:.4f}")

            # Optionally, print stats from trackers
            if self.loss_tracker:
                print("Loss stats:", self.loss_tracker.get_stats())
            if self.forgetting_tracker:
                print("Forgetting stats:", self.forgetting_tracker.get_stats())

    def get_stats(self):
        """Get stats from trackers."""
        stats = {}
        if self.aum_tracker:
            stats['aum_stats'] = self.aum_tracker.get_stats()
        if self.data_map_tracker:
            stats['data_map_stats'] = self.data_map_tracker.get_stats()
        if self.forgetting_tracker:
            stats['forgetting_stats'] = self.forgetting_tracker.get_stats()
<<<<<<< HEAD
        return stats
=======
        if self.loss_tracker:
            stats['loss_stats'] = self.loss_tracker.get_stats()

        stats['predictions'] = self.predictions
        stats['true_labels'] = self.true_labels
        return stats
>>>>>>> origin
