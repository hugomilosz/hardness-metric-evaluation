# import numpy as np
# import torch

# from methods import AumTracker, DataMapTracker, EL2NTracker, ForgettingTracker, GrandTracker, LossTracker
# from transformers import Trainer

# class CustomTrainer(Trainer):
#     def __init__(self, *args, methods=None, num_classes=3, device=None, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.methods = methods or []
#         self.num_classes = num_classes
#         self.total_samples = len(self.train_dataset)

#         # Initialise trackers based on methods
#         self.aum_tracker = AumTracker(self.num_classes) if "aum" in self.methods else None
#         self.data_map_tracker = DataMapTracker(self.total_samples) if "datamaps" in self.methods else None
#         self.el2n_tracker = EL2NTracker(self.total_samples) if "el2n" in self.methods else None
#         self.forgetting_tracker = ForgettingTracker(self.total_samples) if "forgetting" in self.methods else None
#         self.grand_tracker = GrandTracker(self.total_samples) if "grand" in self.methods else None
#         self.loss_tracker = LossTracker(self.total_samples) if "loss" in self.methods else None

#         self.predictions = []
#         self.true_labels = [None] * self.total_samples

#         # Epoch tracking
#         self.current_epoch = 0
#         self.device = device

#     def get_train_dataloader(self):
#         """
#         Returns the training DataLoader, with dataset indices included in each batch.
#         """
#         if self.train_dataset is None:
#             raise ValueError("Trainer: training requires a train_dataset.")

#         train_sampler = self._get_train_sampler() if not isinstance(self.train_dataset, torch.utils.data.IterableDataset) else None

#         def collate_fn(batch):
#             """Extract indices from dataset items."""
#             batch_dict = self.data_collator([{k: v for k, v in item.items() if k != "idx"} for item in batch])
#             batch_indices = [item["idx"] for item in batch]
#             batch_dict['idx'] = batch_indices
#             return batch_dict

#         return torch.utils.data.DataLoader(
#             self.train_dataset,
#             batch_size=self.args.train_batch_size,
#             sampler=train_sampler,
#             collate_fn=collate_fn,
#             drop_last=self.args.dataloader_drop_last,
#             num_workers=self.args.dataloader_num_workers,
#             pin_memory=self.args.dataloader_pin_memory,
#         )


#     def training_step(self, model, inputs, num_items_in_batch=None):
#         """Track per-sample losses and predictions during training step"""
#         dataset_indices = inputs['idx']
#         model_inputs = {key: value.to(self.device) for key, value in inputs.items() if key != 'idx'}
#         model_inputs['output_hidden_states'] = True
#         loss = super().training_step(model, model_inputs, num_items_in_batch)

#         if dataset_indices is not None:
#             dataset_indices = [idx.item() for idx in dataset_indices]

#         if self.methods:
#             with torch.enable_grad():  # Enable gradients
#                 outputs = model(**model_inputs)
#                 logits = outputs.logits
#                 labels = model_inputs["labels"]

#                 predictions = torch.argmax(logits, dim=-1).cpu().numpy()
#                 labels_cpu = labels.cpu().numpy()

#                 # Update predictions
#                 while len(self.predictions) <= self.current_epoch:
#                     self.predictions.append([None] * self.total_samples)
#                 for i, idx in enumerate(dataset_indices):
#                     self.predictions[self.current_epoch][idx] = predictions[i]
#                     if self.true_labels[idx] is None:
#                         self.true_labels[idx] = labels_cpu[i]

#                 # Detach outputs for trackers that don't need gradients
#                 detached_logits = logits.detach()
#                 detached_probs = torch.nn.functional.softmax(detached_logits, dim=-1)

#                 # Update trackers
#                 if self.aum_tracker:
#                     self.aum_tracker.update(detached_logits, labels, dataset_indices)
#                 if self.data_map_tracker:
#                     self.data_map_tracker.update(dataset_indices, detached_logits, labels, detached_probs)
#                 if self.el2n_tracker:
#                     self.el2n_tracker.update(dataset_indices, detached_probs, labels)
#                 if self.forgetting_tracker:
#                     correct_predictions = (predictions == labels_cpu)
#                     self.forgetting_tracker.update(correct_predictions, dataset_indices)
#                 if self.loss_tracker:
#                     self.loss_tracker.update(logits=detached_logits, labels=labels, dataset_indices=dataset_indices)
#                 if self.grand_tracker:
#                     self.grand_tracker.update(dataset_indices, logits, labels, self.model) 

#         return loss

#     def evaluate(self, *args, **kwargs):
#         if self.loss_tracker:
#             self.loss_tracker.finalise_epoch()
#         if self.data_map_tracker:
#             self.data_map_tracker.finalise_epoch()
#         if self.aum_tracker:
#             self.aum_tracker.finalise_epoch()
#         if self.el2n_tracker:
#             self.el2n_tracker.finalise_epoch()
#         if self.grand_tracker:
#             self.grand_tracker.finalise_epoch()
#         self.current_epoch += 1
#         return super().evaluate(*args, **kwargs)

#     def get_unified_stats(self):
#         """Return stats for enabled tracking methods"""
#         stats = {}
#         if self.aum_tracker:
#             stats['aum_stats'] = self.aum_tracker.get_stats()
#         if self.data_map_tracker:
#             stats['data_map_stats'] = self.data_map_tracker.get_stats()
#         if self.el2n_tracker:
#             stats["el2n_scores"] = self.el2n_tracker.get_scores()
#         if self.forgetting_tracker:
#             stats['forgetting_stats'] = self.forgetting_tracker.get_stats()
#         if self.grand_tracker:
#             stats["grand_scores"] = self.grand_tracker.get_scores()
#         if self.loss_tracker:
#             stats['loss_stats'] = self.loss_tracker.get_stats()

#         stats['predictions'] = self.predictions
#         stats['true_labels'] = self.true_labels
#         return stats

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import Trainer
from tqdm import tqdm

from methods import AumTracker, DataMapTracker, EL2NTracker, ForgettingTracker, GrandTracker, LossTracker

class IndependentTrainer:
    def __init__(self, model, train_dataset, eval_dataset=None, methods=None, num_classes=3, device=None, args=None):
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.methods = methods or []
        self.num_classes = num_classes
        self.total_samples = len(self.train_dataset)
        self.device = device
        self.args = args
        
        # Initialize trackers
        self.aum_tracker = AumTracker(self.num_classes) if "aum" in self.methods else None
        self.data_map_tracker = DataMapTracker(self.total_samples) if "datamaps" in self.methods else None
        self.el2n_tracker = EL2NTracker(self.total_samples) if "el2n" in self.methods else None
        self.forgetting_tracker = ForgettingTracker(self.total_samples) if "forgetting" in self.methods else None
        self.grand_tracker = GrandTracker(self.total_samples) if "grand" in self.methods else None
        self.loss_tracker = LossTracker(self.total_samples) if "loss" in self.methods else None
        
        self.predictions = []
        self.true_labels = [None] * self.total_samples
        self.current_epoch = 0
        
        # Define optimizer and loss function
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate)
        self.loss_fn = torch.nn.CrossEntropyLoss()

        torch.autograd.set_detect_anomaly(True)
    
    def get_dataloader(self, dataset, batch_size, shuffle=True):
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=self.args.dataloader_num_workers)
    
    def train(self, num_epochs):
        self.model.train()
        train_dataloader = self.get_dataloader(self.train_dataset, self.args.train_batch_size)
        
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0.0
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)

            for batch in progress_bar:
                dataset_indices = batch["idx"]
                model_inputs = {key: value.to(self.device) for key, value in batch.items() if key != "idx"}

                self.optimizer.zero_grad()
                outputs = self.model(**model_inputs)
                logits = outputs.logits
                labels = model_inputs["labels"]

                loss = torch.nn.functional.cross_entropy(logits, labels)
                loss.backward(retain_graph=True)
                self.optimizer.step()

                total_loss += loss.item()

                # Update progress bar
                progress_bar.set_postfix(loss=loss.item())
                self.track_metrics(logits, labels, dataset_indices)

            print(f"Epoch {epoch + 1}/{num_epochs}, Avg Loss: {total_loss / len(train_dataloader):.4f}")
            self.finalise_epoch()
    
    def track_metrics(self, logits, labels, dataset_indices):
        predictions = torch.argmax(logits, dim=-1).cpu().numpy()
        labels_cpu = labels.cpu().numpy()
        
        while len(self.predictions) <= self.current_epoch:
            self.predictions.append([None] * self.total_samples)
        for i, idx in enumerate(dataset_indices):
            self.predictions[self.current_epoch][idx] = predictions[i]
            if self.true_labels[idx] is None:
                self.true_labels[idx] = labels_cpu[i]
        
        detached_logits = logits.detach()
        detached_probs = torch.nn.functional.softmax(detached_logits, dim=-1)
        
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
        if self.grand_tracker:
            logits = logits.clone().requires_grad_(True)
            self.grand_tracker.update(dataset_indices, logits, labels, self.model)
    
    def finalise_epoch(self):
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
    
    def evaluate(self):
        self.model.eval()
        eval_dataloader = self.get_dataloader(self.eval_dataset, self.args.eval_batch_size, shuffle=False)
        
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in eval_dataloader:
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                labels = inputs['labels']
                outputs = self.model(**inputs)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        accuracy = correct / total
        print(f"Evaluation Accuracy: {accuracy:.4f}")
        return accuracy
    
    def get_unified_stats(self):
        stats = {}
        if self.aum_tracker:
            stats['aum_stats'] = self.aum_tracker.get_stats()
        if self.data_map_tracker:
            stats['data_map_stats'] = self.data_map_tracker.get_stats()
        if self.el2n_tracker:
            stats['el2n_scores'] = self.el2n_tracker.get_scores()
        if self.forgetting_tracker:
            stats['forgetting_stats'] = self.forgetting_tracker.get_stats()
        if self.grand_tracker:
            stats['grand_scores'] = self.grand_tracker.get_scores()
        if self.loss_tracker:
            stats['loss_stats'] = self.loss_tracker.get_stats()
        
        stats['predictions'] = self.predictions
        stats['true_labels'] = self.true_labels
        return stats
