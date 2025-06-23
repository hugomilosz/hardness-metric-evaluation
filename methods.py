import numpy as np
import torch
import torch.nn.functional as F

class AumTracker:
    def __init__(self, total_samples, num_classes):
        """
        A tracker for the Area Under the Margin (AUM) score.

        Args:
            total_samples (int): Total number of samples in the dataset.
            num_classes (int): Number of output classes.
        """
        self.num_classes = num_classes
        self.total_samples = total_samples
        self.epoch_margins = np.full(total_samples, np.nan)
        self.aum_scores = []  # List of arrays, each of shape [total_samples]
    
    def update(self, logits, labels, sample_ids):
        """
        Update the AUM scores based on model predictions and true labels.

        Args:
            logits (torch.Tensor): Model predictions (logits).
            labels (torch.Tensor): Ground truth labels.
            sample_ids (list[int]): Sample indices in the dataset.
        """
        with torch.no_grad():
            logits_np = logits.numpy()
            labels_np = labels.numpy() if isinstance(labels, torch.Tensor) else labels

            for i, sample_id in enumerate(sample_ids):
                sample_logits = logits_np[i]
                label = labels_np[i]
                assigned_logit = sample_logits[label]
                other_logits = np.delete(sample_logits, label)
                largest_other_logit = np.max(other_logits)
                margin = assigned_logit - largest_other_logit
                self.epoch_margins[sample_id] = margin

    def finalise_epoch(self):
        # Save current margins and reset for next epoch
        self.aum_scores.append(self.epoch_margins.copy())
        self.epoch_margins[:] = np.nan

    def get_stats(self):
        """
        Returns:
            np.ndarray: AUM scores as [num_epochs][num_samples]
        """
        return np.stack(self.aum_scores, axis=0)

class DataMapTracker:
    def __init__(self, total_samples):
        """
        Tracks data maps by storing confidence, variability, and correctness per sample across epochs.
        Args:
            total_samples: The total number of samples in the dataset
        """
        self.total_samples = total_samples
        self.gold_label_probs = None  # Will store gold label probabilities (samples, epochs)
        self.epoch_gold_probs = np.full(total_samples, np.nan)  # Current epoch's gold label probs
        self.epoch_sum_probs = np.zeros(total_samples)  # Sum of probs for averaging
        self.seen_counts = np.zeros(total_samples)  # Track samples seen in current epoch
        self.current_epoch = 0
    
    def update(self, dataset_indices, logits, labels, probabilities):
        """
        Updates the data map statistics for the given batch.
        Args:
            dataset_indices: Indices of samples in the dataset
            logits: Model logits
            labels: Ground truth labels
            probabilities: Softmax probabilities from the model
        """
        with torch.no_grad():
            batch_probs = probabilities
            batch_labels = labels
            
            gold_probs = batch_probs[np.arange(len(batch_labels)), batch_labels]
            np.add.at(self.epoch_sum_probs, dataset_indices, gold_probs)
            np.add.at(self.seen_counts, dataset_indices, 1)
    
    def finalise_epoch(self):
        """
        Finalises stats for the current epoch
        """
        # Calculate average gold probability for this epoch
        seen_mask = self.seen_counts > 0
        
        # Divide sum by count to get average probability
        self.epoch_gold_probs[:] = np.nan
        self.epoch_gold_probs[seen_mask] = self.epoch_sum_probs[seen_mask] / self.seen_counts[seen_mask]
        
        # Store this epoch's probabilities
        if self.gold_label_probs is None:
            # Initialize on first epoch
            self.gold_label_probs = np.full((self.total_samples, 1), np.nan)
            self.gold_label_probs[seen_mask, 0] = self.epoch_gold_probs[seen_mask]
        else:
            # Add new column for this epoch
            new_column = np.full((self.total_samples, 1), np.nan)
            new_column[seen_mask, 0] = self.epoch_gold_probs[seen_mask]
            self.gold_label_probs = np.column_stack([self.gold_label_probs, new_column])
        
        # Reset for next epoch
        self.epoch_gold_probs[:] = np.nan
        self.epoch_sum_probs[:] = 0
        self.seen_counts[:] = 0
        self.current_epoch += 1
    
    @property
    def confidence(self):
        """
        Returns:
            Average predictive confidence across epochs: np.array(n_samples)
        """
        if self.gold_label_probs is None:
            return []
        num_epochs = self.gold_label_probs.shape[1]
        return [
            np.nanmean(self.gold_label_probs[:, :epoch_idx + 1], axis=1)
            for epoch_idx in range(num_epochs)
        ]
    
    @property
    def variability(self):
        """
        Returns:
            Epistemic variability of true label probability across epochs: np.array(n_samples)
        """
        if self.gold_label_probs is None:
            return []
        num_epochs = self.gold_label_probs.shape[1]
        return [
            np.nanstd(self.gold_label_probs[:, :epoch_idx + 1], axis=1)
            for epoch_idx in range(num_epochs)
        ]
    
    @property
    def correctness(self):
        """
        Returns:
            Proportion of times a sample is predicted correctly across epochs: np.array(n_samples)
        """
        if self.gold_label_probs is None:
            return []
        num_epochs = self.gold_label_probs.shape[1]
        return [
            np.nanmean(self.gold_label_probs[:, :epoch_idx + 1] > 0.5, axis=1)
            for epoch_idx in range(num_epochs)
        ]
    
    def get_stats(self):
        """
        Returns data map statistics across all epochs.
        """
        if self.gold_label_probs is None or self.gold_label_probs.shape[1] == 0:
            return {"confidence": [], "variability": [], "correctness": []}
        
        return {
            "confidence": self.confidence,
            "variability": self.variability,
            "correctness": self.correctness
        }

class EL2NTracker:
    def __init__(self, total_samples):
        self.total_samples = total_samples
        # Store all epoch scores with shape [num_epochs][num_samples]
        self.el2n_scores = []
        # Initialize the scores as np.nan to represent uncomputed values
        self.current_epoch_scores = np.full((self.total_samples,), np.nan)
    
    def update(self, dataset_indices, probabilities, labels):
        """
        Update the EL2N scores for the current batch.
        
        Args:
            dataset_indices (list[int]): Indices of samples in the dataset
            probabilities (np.ndarray): Softmax probabilities of shape [batch_size, num_classes]
            labels (np.ndarray): Ground truth labels of shape [batch_size]
        """
        # Ensure the input probabilities are in a NumPy array
        batch_probs = probabilities
        batch_labels = labels

        # Ensure one-hot encoding is done correctly
        one_hot_labels = np.zeros_like(batch_probs)
        one_hot_labels[np.arange(batch_labels.size), batch_labels] = 1  

        # Compute EL2N score: L2 norm between probabilities and true one-hot labels
        batch_scores = np.linalg.norm(batch_probs - one_hot_labels, ord=2, axis=1)

        # Store the EL2N scores in the corresponding indices for the current epoch
        for i, idx in enumerate(dataset_indices):
            self.current_epoch_scores[idx] = batch_scores[i]

    def finalise_epoch(self):
        """
        Finalise the scores for the current epoch and reset for the next epoch.
        """
        self.el2n_scores.append(self.current_epoch_scores.copy())
        self.current_epoch_scores[:] = np.nan

    def get_stats(self):
        """
        Returns the EL2N scores across all epochs.
        
        Returns:
            np.ndarray: EL2N scores in shape [num_epochs][num_samples]
        """
        return np.array(self.el2n_scores)


from functorch import make_functional_with_buffers
from torch.func import vmap, grad


class GrandTracker:
    def __init__(self, model, classifier_module, total_samples):
        self.total_samples = total_samples
        self.grand_scores = []
        self.current_epoch_scores = np.full(total_samples, np.nan)

        self.classifier_module = classifier_module
        self.fmodel, self.params, self.buffers = make_functional_with_buffers(self.classifier_module)

    def _compute_per_example_grads(self, features, labels, device):
        params = tuple(p.to(device) for p in self.params)
        buffers = tuple(b.to(device) for b in self.buffers)

        def compute_loss(p, b, x, y):
            logits = self.fmodel(p, b, x.unsqueeze(0))[0]  # [C]
            loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0), y.unsqueeze(0))
            return loss

        grad_fn = grad(compute_loss)
        per_sample_grads = vmap(grad_fn, in_dims=(None, None, 0, 0))(params, buffers, features, labels)
        return per_sample_grads

    def update(self, model_inputs, dataset_indices, device):
        features = model_inputs["features"].to(device)
        labels = model_inputs["labels"].to(device)

        per_example_grads = self._compute_per_example_grads(features, labels, device)

        # Compute L2 norm of per-sample gradients
        squared_norms = sum(g.pow(2).flatten(start_dim=1).sum(dim=1) for g in per_example_grads)
        grad_norms = squared_norms.sqrt().detach().cpu().numpy()  # shape: [batch_size]
        self.current_epoch_scores[np.array(dataset_indices)] = grad_norms

    def finalise_epoch(self):
        self.grand_scores.append(self.current_epoch_scores.copy())
        self.current_epoch_scores = np.full(self.total_samples, np.nan)

    def get_stats(self):
        return np.array(self.grand_scores)


class ForgettingTracker:
    def __init__(self, total_samples):
        self.total_samples = total_samples
        self.last_correct = {i: False for i in range(total_samples)}
        self.forgetting_events = {i: -1 for i in range(total_samples)}
        self.history = []

    def update(self, correct_predictions, sample_indices):
        """Update forgetting events for a batch"""
        for is_correct, sample_idx in zip(correct_predictions, sample_indices):
            if is_correct:
                if self.forgetting_events[sample_idx] == -1:
                    self.forgetting_events[sample_idx] = 0
                if not self.last_correct[sample_idx]:
                    self.last_correct[sample_idx] = True
            else:
                if self.last_correct[sample_idx]:
                    self.forgetting_events[sample_idx] += 1
                    self.last_correct[sample_idx] = False

    def finalise_epoch(self):
        """Save a copy of current forgetting counts (to be called at end of each epoch)"""
        snapshot = np.array([self.forgetting_events[i] for i in range(self.total_samples)])
        self.history.append(snapshot.copy())

    def get_stats(self):
        """Get forgetting statistics"""
        return self.history

class LossTracker:
    def __init__(self, total_samples):
        self.losses = []  # List of loss arrays for each epoch
        self.total_samples = total_samples
        self.current_epoch_losses = {}  # Dictionary to store lists of losses

    def update(self, logits, labels, dataset_indices):
        """Compute per-sample losses for the current batch"""
        labels = torch.as_tensor(labels, device=logits.device)
        losses = torch.nn.functional.cross_entropy(
            logits,
            labels,
            reduction='none'
        ).detach().cpu().numpy()

        for idx, loss in zip(dataset_indices, losses):
            self.current_epoch_losses.setdefault(idx, []).append(loss)

    def finalise_epoch(self):
        """End of epoch - organise losses by sample index"""
        # Use object dtype to store lists in the array
        sorted_losses = np.full(self.total_samples, None, dtype=object)
        
        # Store all losses as lists for each sample
        for idx, losses in self.current_epoch_losses.items():
            sorted_losses[idx] = losses

        self.losses.append(sorted_losses)
        self.current_epoch_losses = {}

    def get_stats(self):
        """Get loss stats - losses[epoch][sample_idx]"""
        return self.losses