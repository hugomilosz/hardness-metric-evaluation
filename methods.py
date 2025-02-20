import torch
import numpy as np

class LossTracker:
    def __init__(self, total_samples):
        self.epoch_losses = []
        self.current_epoch_losses = []
        self.sample_losses = {i: [] for i in range(total_samples)}
        
    def update_batch(self, losses, sample_indices):
        """Update with per-sample losses"""
        for sample_idx, loss in zip(sample_indices, losses):
            self.sample_losses[sample_idx].append(loss)
            self.current_epoch_losses.append(loss)
    
    def update_epoch_loss(self, loss):
        """Update with a single epoch-level loss"""
        self.current_epoch_losses.append(loss)
    
    def finalise_epoch(self):
        """End of epoch"""
        if self.current_epoch_losses:
            epoch_avg_loss = np.mean(self.current_epoch_losses)
            self.epoch_losses.append(epoch_avg_loss)
            self.current_epoch_losses = []
            return epoch_avg_loss
        return None
    
    def get_stats(self):
        """Get loss stats"""
        average_sample_losses = {
            idx: np.mean(losses) if losses else 0.0 
            for idx, losses in self.sample_losses.items()
        }
        return {
            'epoch_losses': self.epoch_losses,
            'per_sample_average_losses': average_sample_losses,
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