import copy
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import Trainer
from tqdm import tqdm

from methods import AumTracker, DataMapTracker, EL2NTracker, ForgettingTracker, GrandTracker, LossTracker

class Trainer:
    def __init__(self, dataset_bundle, methods=None, device=None, args=None):
        self.model = dataset_bundle.model.to(device)
        self.original_model = copy.deepcopy(self.model)
        self.train_dataset = dataset_bundle.train_dataset
        self.eval_dataset = dataset_bundle.eval_dataset
        self.methods = methods or []
        self.num_classes = dataset_bundle.num_labels
        self.total_samples = len(self.train_dataset)
        self.device = device
        self.args = args
        self.dataset_name = dataset_bundle.dataset_name

        self.regularisation_active = False
        self.skip_training = False
        self.num_epochs = self.args.num_train_epochs
        
        # Initialise trackers
        self.aum_tracker = AumTracker(self.num_classes) if "aum" in self.methods else None
        self.data_map_tracker = DataMapTracker(self.total_samples) if "datamaps" in self.methods else None
        self.el2n_tracker = EL2NTracker(self.total_samples) if "el2n" in self.methods else None
        self.forgetting_tracker = ForgettingTracker(self.total_samples) if "forgetting" in self.methods else None
        self.grand_tracker = GrandTracker(self.total_samples) if "grand" in self.methods else None
        self.loss_tracker = LossTracker(self.total_samples) if "loss" in self.methods else None
        if "regularisation" in self.methods:
            self.regularisation_active = True
            if len(methods) == 1:
                self.skip_training = True

        self.misclassified_once = np.zeros(self.total_samples, dtype=bool)
        
        self.true_labels = np.full(self.total_samples, np.nan)
        self.predictions = np.full((self.num_epochs, self.total_samples), np.nan)
        self.current_epoch = 0
        
        # Define optimiser and loss function
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        
        # List to hold binary arrays of misclassified indices
        self.epochwise_misclassified = []

        torch.autograd.set_detect_anomaly(True)
    
    def get_dataloader(self, dataset, batch_size, shuffle=True):
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=self.args.dataloader_num_workers)
    
    def regularisation_stage_train(self):
        """
        Train transformer model for identification stage of JTT.
        Adapted specifically for NLP tasks mentioned in the paper.
        """
        print("Starting JTT Identification Stage Training")
        self.model.train()
        
        # Hyperparameters based on dataset - from the paper
        if self.dataset_name == "multi_nli":
            learning_rate = 0.00002
            l2_reg = 0.0  # No L2 regularization for MultiNLI
        elif self.dataset_name == "civilcomments_wilds":
            learning_rate = 0.00001
            l2_reg = 0.01
        else:  # Default for other NLP datasets
            learning_rate = 0.00002
            l2_reg = 0.01
        
        # Setup AdamW optim with L2 regularisation
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=l2_reg
        )
        
        train_dataloader = self.get_dataloader(self.train_dataset, self.args.train_batch_size)
    
        for epoch in range(self.num_epochs):
            total_loss = 0.0
            epoch_misclassified = np.zeros(self.total_samples, dtype=int)

            progress_bar = tqdm(train_dataloader, desc=f"JTT Identification Epoch {epoch+1}/{self.num_epochs}", leave=True)

            for batch in progress_bar:
                dataset_indices = batch["idx"]
                model_inputs = {key: value.to(self.device) for key, value in batch.items() if key != "idx"}

                optimizer.zero_grad()
                outputs = self.model(**model_inputs)
                logits = outputs.logits
                labels = model_inputs["labels"]

                loss = torch.nn.functional.cross_entropy(logits, labels)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())

                # Track misclassified samples for this epoch
                predictions = torch.argmax(logits, dim=-1).cpu().numpy()
                labels_cpu = labels.cpu().numpy()
                indices_cpu = dataset_indices.cpu().numpy()

                incorrect = predictions != labels_cpu
                epoch_misclassified[indices_cpu[incorrect]] = 1
            print(f"[JTT Identification] Epoch {epoch+1}, Avg Loss: {total_loss / len(train_dataloader):.4f}")
            self.epochwise_misclassified.append(epoch_misclassified)

        # Reset model and optimiser for the second stage of training
        self.model = copy.deepcopy(self.original_model).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.current_epoch = 0
    
    def train(self):
        if self.regularisation_active:
            self.regularisation_stage_train()
        if self.skip_training:
            return
        self.model.train()
        train_dataloader = self.get_dataloader(self.train_dataset, self.args.train_batch_size)
        
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0.0
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}", leave=True)

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
                self.track_metrics(logits, labels, dataset_indices, model_inputs)

            print(f"Epoch {epoch + 1}/{self.num_epochs}, Avg Loss: {total_loss / len(train_dataloader):.4f}")
            self.finalise_epoch()
    
    def track_metrics(self, logits, labels, dataset_indices, model_inputs):
        predictions = torch.argmax(logits, dim=-1).cpu().numpy()
        labels_cpu = labels.cpu().numpy()
        dataset_indices = np.array(dataset_indices)

        # Update predictions for the current epoch
        self.predictions[self.current_epoch][dataset_indices] = predictions

        # Only update unseen true labels
        unseen_mask = np.isnan(self.true_labels[dataset_indices])
        self.true_labels[dataset_indices[unseen_mask]] = labels_cpu[unseen_mask]

        # Continue with tracker updates
        detached_logits = logits.detach()
        detached_probs = torch.nn.functional.softmax(detached_logits, dim=-1)

        if self.aum_tracker:
            self.aum_tracker.update(detached_logits, labels, dataset_indices.tolist())
        if self.data_map_tracker:
            self.data_map_tracker.update(dataset_indices.tolist(), detached_logits, labels, detached_probs)
        if self.el2n_tracker:
            self.el2n_tracker.update(dataset_indices.tolist(), detached_probs, labels)
        if self.forgetting_tracker:
            correct_predictions = (predictions == labels_cpu)
            self.forgetting_tracker.update(correct_predictions, dataset_indices.tolist())
        if self.loss_tracker:
            self.loss_tracker.update(logits=detached_logits, labels=labels, dataset_indices=dataset_indices.tolist())
        if self.grand_tracker:
            if hasattr(self.model, "logits_proj"):
                classifier_module = self.model.logits_proj  # Use logits_proj if it exists
            else:
                classifier_module = self.model.classifier  # Otherwise, use classifier
            classifier_params = list(classifier_module.parameters())

            self.grand_tracker.update(dataset_indices.tolist(), logits, labels, classifier_params)

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
        if self.forgetting_tracker:
            self.forgetting_tracker.finalise_epoch()
        self.current_epoch += 1
    
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
            stats['regularisation'] = self.epochwise_misclassified
        
        stats['predictions'] = self.predictions
        stats['true_labels'] = self.true_labels

        if "accuracy" in self.methods:
            all_accuracies = []
            correct_so_far = np.zeros(len(self.true_labels), dtype=float)
            for epoch_idx, epoch_preds in enumerate(self.predictions):
                correct = (np.array(epoch_preds) == np.array(self.true_labels)).astype(float)
                correct_so_far += correct
                avg_correct = correct_so_far / (epoch_idx + 1)
                all_accuracies.append(avg_correct)
            stats['accuracy'] = all_accuracies

        return stats
