import torch
from transformers import Trainer, get_linear_schedule_with_warmup
import numpy as np
from methods import LossTracker
from methods import ForgettingTracker

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

        if self.loss_tracker:
            self.loss_tracker.finalise_epoch()

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
        if self.loss_tracker:
            stats['loss_stats'] = self.loss_tracker.get_stats()
        if self.forgetting_tracker:
            stats['forgetting_stats'] = self.forgetting_tracker.get_stats()
        return stats
