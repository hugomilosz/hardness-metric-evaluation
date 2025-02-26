import torch
import numpy as np

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

    def get_stats(self):
        """Get loss stats - losses[epoch][sample_idx]"""
        if not self.losses:
            print("[WARNING] No loss data available!")
            return {'epoch_losses': [], 'per_sample_losses': {}}

        epoch_losses = [np.mean(epoch_losses) for epoch_losses in self.losses]
        per_sample_losses = np.array(self.losses).T.tolist()

        return {
            'epoch_losses': epoch_losses,
            'per_sample_losses': {i: losses for i, losses in enumerate(per_sample_losses)}
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