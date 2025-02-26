import torch
import numpy as np

class AumTracker:
    def __init__(self, num_classes, save_dir=".", compressed=True):
        """
        A tracker for the AUM (Area Under the Margin) score.

        Args:
          num_classes (int): The number of classes in the classification task.
          save_dir (str): Directory to save intermediate results (if needed).
          compressed (bool): Whether to use compressed format for saving results.
        """
        self.num_classes = num_classes
        self.save_dir = save_dir
        self.compressed = compressed
        self.margins = []
        self.scores = []

    def update(self, y_pred, y_true, sample_ids):
        """
        Update the AUM scores based on model predictions and true labels.
        
        Args:
          y_pred (torch.Tensor): Model predictions (logits).
          y_true (torch.Tensor): True labels for the batch.
          sample_ids (torch.Tensor): Unique sample IDs for the batch.
        """
        # Compute the margin for each sample (difference between correct class score and others)
        logits = y_pred  # Predicted logits
        true_class_logits = logits.gather(1, y_true.view(-1, 1))  # Get the logits for the true class
        margins = logits - true_class_logits  # Margins for each class
        
        # We need to calculate the area under the margin for each sample.
        for i in range(margins.shape[0]):
            sample_margin = margins[i, :].cpu().numpy()  # Convert margin to numpy
            self.margins.append(sample_margin)
        
    def finalise_epoch(self):
        """
        Finalise the AUM calculation.
        """
        all_margins = np.array(self.margins)
        aum_scores = np.mean(all_margins, axis=1)
        self.scores.append(aum_scores.tolist())
        self.margins = []

    def get_stats(self):
        """
        Return AUM statistics.
        """
        return {"aum_scores": self.scores}

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
                self.seen_counts[idx] += 1
                
                # Get probability assigned to the gold/correct class
                gold_class = batch_labels[i]
                gold_prob = batch_probs[i, gold_class]
                
                # Store the gold probability directly
                self.epoch_gold_probs[idx] = gold_prob
    
    def finalise_epoch(self):
        """
        Finalises stats for the current epoch
        """
        # Only consider samples that were seen (not NaN)
        seen_mask = ~np.isnan(self.epoch_gold_probs)
        
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
        self.seen_counts[:] = 0
        self.current_epoch += 1
    
    @property
    def confidence(self):
        """
        Returns:
            Average predictive confidence across epochs: np.array(n_samples)
        """
        if self.gold_label_probs is None:
            return np.array([])
        return np.nanmean(self.gold_label_probs, axis=1)
    
    @property
    def variability(self):
        """
        Returns:
            Epistemic variability of true label probability across epochs: np.array(n_samples)
        """
        if self.gold_label_probs is None:
            return np.array([])
        return np.nanstd(self.gold_label_probs, axis=1)
    
    @property
    def correctness(self):
        """
        Returns:
            Proportion of times a sample is predicted correctly across epochs: np.array(n_samples)
        """
        if self.gold_label_probs is None:
            return np.array([])
        # Using 0.5 as threshold for correctness, same as reference implementation
        return np.nanmean(self.gold_label_probs > 0.5, axis=1)
    
    def get_stats(self):
        """
        Returns data map statistics across all epochs.
        """
        if self.gold_label_probs is None or self.gold_label_probs.shape[1] == 0:
            return {"confidence": [], "variability": [], "correctness": []}
        
        # Calculate metrics across epochs
        confidence = np.nanmean(self.gold_label_probs, axis=1)  # Average confidence
        variability = np.nanstd(self.gold_label_probs, axis=1)  # Standard deviation across epochs
        correctness = np.nanmean(self.gold_label_probs > 0.5, axis=1)  # Proportion correct
        
        return {
            "confidence": confidence,
            "variability": variability,
            "correctness": correctness
        }

class ForgettingTracker:
    def __init__(self, total_samples):
        self.last_correct = {i: False for i in range(total_samples)}
        self.forgetting_events = {i: -1 for i in range(total_samples)}
    
    def update(self, correct_predictions, sample_indices):
        """Update forgetting events for a batch"""
        for is_correct, sample_idx in zip(correct_predictions, sample_indices):
            if is_correct:
                # first time learnt
                if self.forgetting_events[sample_idx] == -1:
                    self.forgetting_events[sample_idx] = 0
                if not self.last_correct[sample_idx]:
                    self.last_correct[sample_idx] = True
            else:
                # forgotten
                if self.last_correct[sample_idx]:
                    self.forgetting_events[sample_idx] += 1
                    self.last_correct[sample_idx] = False
    
    def get_stats(self):
        """Get forgetting statistics"""
        forgettable_examples = {
            idx: count for idx, count in self.forgetting_events.items()
            if count > 0 or count == -1
        }
        
        forgetting_values = [count for count in self.forgetting_events.values() if count >= 0]
        
        return {
            'forgetting_events': self.forgetting_events,
            'forgettable_examples': forgettable_examples,
        }

class LossTracker:
    def __init__(self, total_samples):
        self.losses = []
        self.total_samples = total_samples
        self.current_epoch_losses = []

    def update(self, logits, labels, dataset_indices):
        """Compute per-sample losses for the current batch"""
        batch_losses = []
        for i in range(len(logits)):
            loss = torch.nn.functional.cross_entropy(
                logits[i].unsqueeze(0),
                labels[i].unsqueeze(0),
                reduction='none'
            )
            batch_losses.append(loss.detach().cpu().item())
            
        # Store the losses with their indices
        for idx, loss in zip(dataset_indices, batch_losses):
            self.current_epoch_losses.append((idx, loss))
            
        return np.mean(batch_losses)

    def finalise_epoch(self):
        """End of epoch - organize losses by sample index"""
        # Sort losses by sample index
        sorted_losses = [0.0] * self.total_samples
        for idx, loss in self.current_epoch_losses:
            sorted_losses[idx] = loss
            
        self.losses.append(sorted_losses)
        self.current_epoch_losses = []  # Reset for next epoch
        
        return np.mean(sorted_losses)

    def get_stats(self):
        """Get loss stats - losses[epoch][sample_idx]"""
        epoch_losses = [np.nanmean(epoch_losses) for epoch_losses in self.losses]
        # Transpose losses to get per-sample trajectories
        per_sample_losses = np.array(self.losses).T.tolist()
        
        return {
            'epoch_losses': epoch_losses,
            'per_sample_losses': {i: losses for i, losses in enumerate(per_sample_losses)},
            'all_losses': self.losses
        }