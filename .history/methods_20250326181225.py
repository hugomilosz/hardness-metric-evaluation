import numpy as np
import torch

class LossTracker:
    def __init__(self, total_samples):
        self.losses = []
        self.total_samples = total_samples
        self.current_epoch_losses = []

    def update_from_batch(self, logits, labels, dataset_indices):
        """Compute per-sample losses for the current batch"""
        if dataset_indices is None or len(dataset_indices) == 0:
            print("[ERROR] No dataset indices provided!")
            return 0  # Return 0 loss to prevent crashes

        batch_losses = []
        for i in range(len(logits)):
            loss = torch.nn.functional.cross_entropy(
                logits[i].unsqueeze(0),
                labels[i].unsqueeze(0),
                reduction='none'
            )
            batch_losses.append(loss.detach().cpu().item())

        # Store the losses with their dataset indices
        for idx, loss in zip(dataset_indices, batch_losses):
            # print(f"[DEBUG] Storing loss for idx {idx}: {loss}")  # Debug print
            self.current_epoch_losses.append((idx, loss))

        return np.mean(batch_losses)

    def finalise_epoch(self):
        """End of epoch - organize losses by sample index"""
        if len(self.current_epoch_losses) == 0:
            print("[WARNING] No losses recorded for this epoch!")
            return 0  # Return 0 to indicate no losses were stored

        # Sort losses by sample index
        sorted_losses = [0] * self.total_samples
        for idx, loss in self.current_epoch_losses:
            if sorted_losses[idx] != 0:
                print(f"[WARNING] Duplicate index {idx}, overwriting loss.")
            sorted_losses[idx] = loss

        self.losses.append(sorted_losses)
        self.current_epoch_losses = []  # Reset for next epoch

        return np.mean(sorted_losses)

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
        """Get loss stats - losses[epoch][sample_idx]"""
        if not self.losses:
            print("[WARNING] No loss data available!")
            return {'epoch_losses': [], 'per_sample_losses': {}}

        epoch_losses = [np.mean(epoch_losses) for epoch_losses in self.losses]
        per_sample_losses = np.array(self.losses).T.tolist()

        return {
            "aum_scores": aum_scores,
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
        self.epoch_gold_probs[:] = np.nan  # Default for unseen samples
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
        return np.nanmean(self.gold_label_probs > 0.5, axis=1)
    
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
        self.current_epoch_losses_sum = {}
        self.current_epoch_counts = {}

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
            
        # Store the losses with their indices, accumulating for duplicates
        for idx, loss in zip(dataset_indices, batch_losses):
            if idx not in self.current_epoch_losses_sum:
                self.current_epoch_losses_sum[idx] = loss
                self.current_epoch_counts[idx] = 1
            else:
                self.current_epoch_losses_sum[idx] += loss
                self.current_epoch_counts[idx] += 1
            
        return np.mean(batch_losses)

    def finalise_epoch(self):
        """End of epoch - organise losses by sample index"""
        # Sort losses by sample index
        sorted_losses = [float('nan')] * self.total_samples
        for idx in self.current_epoch_losses_sum:
            avg_loss = self.current_epoch_losses_sum[idx] / self.current_epoch_counts[idx]
            sorted_losses[idx] = avg_loss
            
        self.losses.append(sorted_losses)

        self.current_epoch_losses_sum = {}
        self.current_epoch_counts = {}
        
        # Calculate epoch mean (ignoring NaN values)
        valid_losses = [loss for loss in sorted_losses if not np.isnan(loss)]
        epoch_mean = np.mean(valid_losses) if valid_losses else float('nan')
        
        return epoch_mean

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