import copy
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import Trainer
from tqdm import tqdm

from methods import AumTracker, DataMapTracker, EL2NTracker, ForgettingTracker, GrandTracker, LossTracker

class IndependentTrainer:
    def __init__(self, model, train_dataset, eval_dataset=None, methods=None, num_classes=3, device=None, args=None):
        self.original_model = copy.deepcopy(model)
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.methods = methods or []
        self.num_classes = num_classes
        self.total_samples = len(self.train_dataset)
        self.device = device
        self.args = args
        
        # Initialise trackers
        self.aum_tracker = AumTracker(self.num_classes) if "aum" in self.methods else None
        self.data_map_tracker = DataMapTracker(self.total_samples) if "datamaps" in self.methods else None
        self.el2n_tracker = EL2NTracker(self.total_samples) if "el2n" in self.methods else None
        self.forgetting_tracker = ForgettingTracker(self.total_samples) if "forgetting" in self.methods else None
        self.grand_tracker = GrandTracker(self.total_samples) if "grand" in self.methods else None
        self.loss_tracker = LossTracker(self.total_samples) if "loss" in self.methods else None
        self.regularisation_active = "regularisation" in self.methods

        self.misclassified_once = np.zeros(self.total_samples, dtype=bool)
        
        self.predictions = []
        self.true_labels = [None] * self.total_samples
        self.current_epoch = 0
        
        # Define optimizer and loss function
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate)
        self.loss_fn = torch.nn.CrossEntropyLoss()

        torch.autograd.set_detect_anomaly(True)
    
    def get_dataloader(self, dataset, batch_size, shuffle=True):
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=self.args.dataloader_num_workers)
    
    def regularisation_stage_train(self, num_epochs=3, dropout_rate=0.3, l1_lambda=1e-5):
        """
        Train BERT with high dropout and L1 regularisation to identify hard/easy examples. 
        Misclassified examples are considered 'hard'.
        """
        self.model.train()

        # Apply dropout
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = dropout_rate

        train_dataloader = self.get_dataloader(self.train_dataset, self.args.train_batch_size)
        for epoch in range(num_epochs):
            total_loss = 0.0
            progress_bar = tqdm(train_dataloader, desc=f"Regularisation Epoch {epoch+1}/{num_epochs}", leave=True)
            for batch in progress_bar:
                dataset_indices = batch["idx"]
                model_inputs = {key: value.to(self.device) for key, value in batch.items() if key != "idx"}

                self.optimizer.zero_grad()
                outputs = self.model(**model_inputs)
                logits = outputs.logits
                labels = model_inputs["labels"]

                # Cross-entropy + L1 regularisation
                ce_loss = torch.nn.functional.cross_entropy(logits, labels)
                l1_penalty = sum(param.abs().sum() for param in self.model.parameters())
                loss = ce_loss + l1_lambda * l1_penalty

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())
                self.track_metrics(logits, labels, dataset_indices)

            print(f"[Regularisation] Epoch {epoch+1}, Avg Loss: {total_loss / len(train_dataloader):.4f}")
            self.finalise_epoch()
            # self.current_epoch += 1
        print("Regularisation training complete")

        # Reset model and optimiser
        self.model = copy.deepcopy(self.original_model).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
    
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
            # self.current_epoch += 1
    
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

        if self.regularisation_active:
            correct_predictions = (predictions == labels_cpu)
            for i, idx in enumerate(dataset_indices):
                if not correct_predictions[i]:
                    self.misclassified_once[idx] = True
            return
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
            stats['aum'] = self.aum_tracker.get_stats()
        if self.data_map_tracker:
            stats['datamap'] = self.data_map_tracker.get_stats()
        if self.el2n_tracker:
            stats['el2n'] = self.el2n_tracker.get_scores()
        if self.forgetting_tracker:
            stats['forgetting'] = self.forgetting_tracker.get_stats()
        if self.grand_tracker:
            stats['grand'] = self.grand_tracker.get_scores()
        if self.loss_tracker:
            stats['loss'] = self.loss_tracker.get_stats()
        if self.regularisation_active:
            stats['regularisation'] = self.misclassified_once.astype(int).tolist()
        
        stats['predictions'] = self.predictions
        stats['true_labels'] = self.true_labels
        return stats
