import torch
import numpy as np

class LossTracker:
    def __init__(self, total_samples):
        self.per_epoch_sample_losses = {i: [] for i in range(total_samples)}
        self.current_epoch_losses = []
        self.epoch_losses = []
        self.total_samples = total_samples

    def update_batch(self, losses, sample_indices):
        """Update with per-sample losses"""
        for sample_idx, loss in zip(sample_indices, losses):
            # Only append if we haven't recorded a loss for this sample in the current epoch
            if len(self.per_epoch_sample_losses[sample_idx]) < len(self.epoch_losses) + 1:
                self.per_epoch_sample_losses[sample_idx].append(loss)
            else:
                # Update the current epoch's loss if we've seen this sample again
                self.per_epoch_sample_losses[sample_idx][-1] = loss

    def update_epoch_loss(self, loss):
        """Update with a single epoch-level loss"""
        self.current_epoch_losses.append(loss)

    def update_from_batch(self, logits, labels, dataset_indices):
        """Compute per-sample losses and update tracking"""
        per_sample_loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        per_sample_losses = per_sample_loss_fct(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )

        # Update per-sample losses with the current batch
        self.update_batch(
            [loss.item() for loss in per_sample_losses],
            dataset_indices
        )
        
        batch_loss = per_sample_losses.mean().detach().cpu().item()
        return batch_loss

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
        return {
            'epoch_losses': self.epoch_losses,
            'per_sample_losses': self.per_epoch_sample_losses,
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