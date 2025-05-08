import numpy as np
import torch
import torch.nn.functional as F

class AumTracker:
    def __init__(self, num_classes):
        """
        A tracker for the Area Under the Margin (AUM) score.
        
        Args:
            num_classes (int): The number of classes in the classification task.
        """
        self.num_classes = num_classes
        self.sample_margins = {}
        self.epoch_margins = {}
        self.current_epoch = 0
        
    def update(self, logits, labels, sample_ids):
        """
        Update the AUM scores based on model predictions and true labels.
        
        Args:
            logits (torch.Tensor): Model predictions (logits, pre-softmax outputs).
            labels (torch.Tensor): True labels for the batch.
            sample_ids (list): Unique sample IDs for the batch.
        """
        with torch.no_grad():
            logits_np = logits.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
            
            for i, sample_id in enumerate(sample_ids):
                sample_id = int(sample_id)
                sample_logits = logits_np[i]
                assigned_class = labels_np[i]
                assigned_logit = sample_logits[assigned_class]

                other_logits = np.delete(sample_logits, assigned_class)
                largest_other_logit = np.max(other_logits)
                margin = assigned_logit - largest_other_logit
                self.epoch_margins[sample_id] = margin
    
    def finalise_epoch(self):
        """
        Finalise the AUM calculation for the current epoch and prepare for the next.
        """
        # Transfer epoch margins to cumulative tracking
        for sample_id, margin in self.epoch_margins.items():
            if sample_id not in self.sample_margins:
                self.sample_margins[sample_id] = []
            self.sample_margins[sample_id].append(margin)
        
        # Clear epoch margins for next epoch
        self.epoch_margins = {}
        self.current_epoch += 1
    
    def get_stats(self):
        """
        Calculate and return AUM statistics for all samples across all epochs.
        
        Returns:
            dict: Dictionary with sample IDs and their AUM scores
        """
        aum_scores = {}
        for sample_id, margins in self.sample_margins.items():
            # AUM is the average margin over all epochs
            aum_scores[sample_id] = sum(margins) / len(margins)
        
        return {
            # "aum_scores": aum_scores,
            "sample_margins": self.sample_margins
        }

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
            batch_probs = probabilities.cpu().numpy()
            batch_labels = labels.cpu().numpy()
            
            for i, idx in enumerate(dataset_indices):
                # Get probability assigned to the gold/correct class
                gold_class = batch_labels[i]
                gold_prob = batch_probs[i, gold_class]
                
                # Accumulate probabilities for averaging at end of epoch
                self.epoch_sum_probs[idx] += gold_prob
                self.seen_counts[idx] += 1
    
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
        self.el2n_scores = []
        self.current_epoch_scores = np.full(self.total_samples, np.nan)

    def update(self, dataset_indices, probabilities, labels):
        with torch.no_grad():
            batch_probs = probabilities.cpu().numpy()
            batch_labels = labels.cpu().numpy()
            num_classes = batch_probs.shape[1]

            # Ensure one-hot encoding is done correctly
            one_hot_labels = np.zeros_like(batch_probs)
            one_hot_labels[np.arange(batch_labels.size), batch_labels] = 1  

            # Compute EL2N score: L2 norm between probabilities and true one-hot labels
            batch_scores = np.linalg.norm(batch_probs - one_hot_labels, ord=2, axis=1)

            # Store the scores
            for i, idx in enumerate(dataset_indices):
                self.current_epoch_scores[idx] = batch_scores[i]

    def finalise_epoch(self):
        # Append the current epoch's EL2N scores
        self.el2n_scores.append(self.current_epoch_scores.copy())

        # Reset for the next epoch
        self.current_epoch_scores = np.full(self.total_samples, np.nan)

    def get_scores(self):
        return np.array(self.el2n_scores)

import torch
import torch.nn.functional as F

class GrandTracker:
    def __init__(self, total_samples):
        self.total_samples = total_samples
        self.grand_scores = []  # To store GraNd scores for each epoch
        self.current_epoch_scores = np.full(total_samples, np.nan)
        
    def update(self, dataset_indices, logits, labels, classifier_params):
        # Compute the loss for each sample individually
        losses = F.cross_entropy(logits, labels, reduction='none')
        
        # For each sample, calculate the gradient of the loss with respect to classifier parameters
        for i, idx in enumerate(dataset_indices):
            loss = losses[i]

            # Compute gradients of the loss w.r.t. classifier parameters
            grad = torch.autograd.grad(
                outputs=loss,  # Single sample loss
                inputs=classifier_params,  # Only classifier layer parameters
                retain_graph=True,  # Do NOT retain the graph after gradient computation
                create_graph=False,  # No need for second-order gradients
                allow_unused=True  # Allow unused parameters
            )
            
            # Compute the gradient norm (L2 norm)
            grad_norm = 0.0
            for g in grad:
                if g is not None:
                    grad_norm += (g**2).sum().item()  # Sum of squared gradients for each parameter

            grad_norm = torch.sqrt(torch.tensor(grad_norm, device=logits.device))  # L2 norm of gradient

            # Store the gradient norm for this sample
            self.current_epoch_scores[idx] = grad_norm.item()
    
    def finalise_epoch(self):
        # Store the current epoch's GraNd and reset
        self.grand_scores.append(self.current_epoch_scores.copy())
        self.current_epoch_scores = np.full(self.total_samples, np.nan)
        
    def get_scores(self):
        # Return all the recorded GraNd scores across epochs
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
        return {
            # 'forgetting_events': self.forgetting_events,
            'epoch_history': self.history,
        }


class LossTracker:
    def __init__(self, total_samples):
        self.losses = []
        self.total_samples = total_samples
        self.current_epoch_losses_sum = {}
        self.current_epoch_counts = {}

    def update(self, logits, labels, dataset_indices):
        """Compute per-sample losses for the current batch"""
        batch_losses = []
        labels = labels.to(logits.device)
        for i in range(len(logits)):
            loss = torch.nn.functional.cross_entropy(
                logits[i].unsqueeze(0),
                labels[i].unsqueeze(0),
                reduction='sum'
            )
            batch_losses.append(loss.detach().cpu().item())
            
        # Store losses with their indices, accumulating for duplicates
        for idx, loss in zip(dataset_indices, batch_losses):
            if idx not in self.current_epoch_losses_sum:
                self.current_epoch_losses_sum[idx] = []
            self.current_epoch_losses_sum[idx].append(loss)

    def finalise_epoch(self):
        """End of epoch - organise losses by sample index"""
        sorted_losses = [np.nan] * self.total_samples
        for idx, losses in self.current_epoch_losses_sum.items():
            sorted_losses[idx] = np.mean(losses)

        self.losses.append(sorted_losses)
        self.current_epoch_losses_sum = {}

    def get_stats(self):
        """Get loss stats - losses[epoch][sample_idx]"""
        # epoch_losses = [np.nanmean(epoch_losses) for epoch_losses in self.losses]
        per_sample_losses = np.array(self.losses).T.tolist()
        return {
            # 'epoch_losses': epoch_losses,
            # 'per_sample_losses': {i: losses for i, losses in enumerate(per_sample_losses)},
            'all_losses': self.losses
        }