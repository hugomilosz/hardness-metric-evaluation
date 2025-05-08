import copy
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluate import load as load_metric
from methods import AumTracker, DataMapTracker, EL2NTracker, ForgettingTracker, GrandTracker, LossTracker
from transformers import get_scheduler, AdamW
from accelerate import Accelerator

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Trainer:
    def __init__(self, dataset_bundle, methods=None, device=None, args=None, eval_metrics=None):
        self.accelerator = Accelerator()
        self.args = args
        self.seed = args.seed
        set_seed(self.seed)

        self.model = dataset_bundle.model
        self.original_model = copy.deepcopy(self.model)
        self.train_dataset = dataset_bundle.train_dataset
        self.eval_dataset = dataset_bundle.eval_dataset
        self.methods = methods or []
        self.num_classes = dataset_bundle.num_labels
        self.total_samples = len(self.train_dataset)
        self.dataset_name = dataset_bundle.dataset_name
        self.num_epochs = self.args.num_train_epochs
        self.eval_metrics = eval_metrics
        self.gradient_accumulation_steps = getattr(args, "gradient_accumulation_steps", 1)
        self.lr_scheduler_type = getattr(args, "lr_scheduler_type", "linear")
        self.num_warmup_steps = args.warmup_ratio

        self.regularisation_active = "regularisation" in self.methods
        self.skip_training = self.regularisation_active and len(methods) == 1

        # Initialise trackers
        self.aum_tracker = AumTracker(self.num_classes) if "aum" in self.methods else None
        self.data_map_tracker = DataMapTracker(self.total_samples) if "datamaps" in self.methods else None
        self.el2n_tracker = EL2NTracker(self.total_samples) if "el2n" in self.methods else None
        self.forgetting_tracker = ForgettingTracker(self.total_samples) if "forgetting" in self.methods else None
        self.grand_tracker = GrandTracker(self.total_samples) if "grand" in self.methods else None
        self.loss_tracker = LossTracker(self.total_samples) if "loss" in self.methods else None

        self.misclassified_once = np.zeros(self.total_samples, dtype=bool)
        self.true_labels = np.full(self.total_samples, np.nan, dtype=np.float32)
        self.predictions = np.full((self.num_epochs, self.total_samples), np.nan)
        self.current_epoch = 0
        self.epochwise_misclassified = []

        if device and not self.accelerator.distributed_type == "MULTI_GPU":
            self.model = self.model.to(device)
            self.device = device
        else:
            self.device = self.accelerator.device

    def get_dataloader(self, dataset, batch_size, shuffle=True):
        generator = torch.Generator()
        generator.manual_seed(self.seed)  # Ensures reproducibility for shuffling

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=True,
            generator=generator
        )

    def prepare_training(self):
        train_dataloader = self.get_dataloader(self.train_dataset, self.args.train_batch_size)
        eval_dataloader = self.get_dataloader(self.eval_dataset, self.args.eval_batch_size, shuffle=False)

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=self.args.weight_decay,
        )

        total_steps = (len(train_dataloader) // self.gradient_accumulation_steps) * self.num_epochs
        warmup_steps = int(self.num_warmup_steps * total_steps) if isinstance(self.num_warmup_steps, float) else self.num_warmup_steps
        self.scheduler = get_scheduler(self.lr_scheduler_type, self.optimizer, warmup_steps, total_steps)

        self.model, self.optimizer, train_dataloader, eval_dataloader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, train_dataloader, eval_dataloader, self.scheduler
        )

        return train_dataloader, eval_dataloader

    def train(self):
        train_dataloader, eval_dataloader = self.prepare_training()
        best_accuracy = 0.0
        best_model_state = None

        # print("Running initial evaluation...")
        # baseline_metrics = self.evaluate(eval_dataloader)
        # print(f"Baseline Evaluation Metrics: {baseline_metrics}")

        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0.0
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}", leave=True)
            self.optimizer.zero_grad()

            for step, batch in enumerate(progress_bar):
                dataset_indices = batch.get("idx", None)
                if dataset_indices is None:
                    print("Warning: 'idx' not found in batch. Check data preprocessing.")
                    continue

                model_inputs = {k: v for k, v in batch.items() if k != "idx"}

                for k, v in model_inputs.items():
                    if isinstance(v, torch.Tensor) and not v.device == self.accelerator.device:
                        model_inputs[k] = v.to(self.accelerator.device)

                outputs = self.model(**model_inputs)
                logits = outputs.logits
                labels = model_inputs["labels"]

                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, self.num_classes),
                    labels.view(-1),
                    reduction="mean"
                )

                loss = loss / self.gradient_accumulation_steps
                self.accelerator.backward(loss, retain_graph=True)

                if (step + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                total_loss += loss.item() * self.gradient_accumulation_steps
                progress_bar.set_postfix(loss=loss.item() * self.gradient_accumulation_steps)

                self.track_metrics(logits, labels, dataset_indices, model_inputs)

            avg_loss = total_loss / len(train_dataloader)
            print(f"Epoch {epoch + 1}/{self.num_epochs}, Avg Loss: {avg_loss:.4f}")
            self.finalise_epoch()

            eval_metrics = self.evaluate(eval_dataloader)
            print(f"Epoch {epoch + 1} Evaluation Metrics: {eval_metrics}")

    def track_metrics(self, logits, labels, dataset_indices, model_inputs):
        predictions = torch.argmax(logits, dim=-1).cpu().numpy()
        labels_cpu = labels.cpu().numpy()
        dataset_indices = dataset_indices.cpu().numpy()

        self.predictions[self.current_epoch][dataset_indices] = predictions
        unseen_mask = np.isnan(self.true_labels[dataset_indices])
        self.true_labels[dataset_indices[unseen_mask]] = labels_cpu[unseen_mask]

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
            # else:
            #     print("Warning: Could not find classifier module for GRAND tracker")

    def finalise_epoch(self):
        for tracker in [self.loss_tracker, self.data_map_tracker, self.aum_tracker,
                        self.el2n_tracker, self.grand_tracker, self.forgetting_tracker]:
            if tracker:
                tracker.finalise_epoch()
        self.current_epoch += 1

    def evaluate(self, eval_dataloader=None):
        self.model.eval()
        if eval_dataloader is None:
            eval_dataloader = self.get_dataloader(self.eval_dataset, self.args.eval_batch_size, shuffle=False)
            eval_dataloader = self.accelerator.prepare(eval_dataloader)

        all_preds, all_labels = [], []
        eval_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                inputs = {k: v for k, v in batch.items() if k != "idx"}
                labels = inputs["labels"]

                outputs = self.model(**inputs)
                logits = outputs.logits

                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, self.num_classes),
                    labels.view(-1),
                    reduction="mean"
                )
                eval_loss += loss.item()

                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(self.accelerator.gather(preds).cpu().tolist())
                all_labels.extend(self.accelerator.gather(labels).cpu().tolist())

        min_len = min(len(all_preds), len(all_labels))
        all_preds = all_preds[:min_len]
        all_labels = all_labels[:min_len]

        metrics = {"loss": eval_loss / len(eval_dataloader)}
        print(f"Evaluation samples: {len(all_labels)}")

        for metric_name in self.eval_metrics:
            try:
                metric = load_metric(metric_name)
                if metric_name in ["f1", "precision", "recall"]:
                    result = metric.compute(predictions=all_preds, references=all_labels, average="macro")
                else:
                    result = metric.compute(predictions=all_preds, references=all_labels)

                if isinstance(result, dict):
                    metrics.update(result)
                else:
                    metrics[metric_name] = result
            except Exception as e:
                print(f"Error computing {metric_name}: {str(e)}")

        for name, score in metrics.items():
            print(f"Eval {name.capitalize()}: {score:.4f}" if isinstance(score, (int, float)) else f"Eval {name.capitalize()}: {score}")

        return metrics

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

        stats['predictions'] = self.predictions
        stats['true_labels'] = self.true_labels

        if "accuracy" in self.methods:
            all_accuracies = []
            correct_so_far = np.zeros(len(self.true_labels), dtype=float)
            for epoch_idx, epoch_preds in enumerate(self.predictions):
                valid_indices = ~np.isnan(epoch_preds) & ~np.isnan(self.true_labels)
                if np.sum(valid_indices) > 0:
                    correct = (epoch_preds[valid_indices] == self.true_labels[valid_indices]).astype(float)
                    correct_so_far[valid_indices] += correct
                    epoch_count = np.zeros_like(correct_so_far) + (epoch_idx + 1)
                    epoch_count[~valid_indices] = 1.0
                    avg_correct = correct_so_far / epoch_count
                    all_accuracies.append(avg_correct)
                else:
                    all_accuracies.append(np.zeros_like(correct_so_far))

            stats['accuracy'] = all_accuracies

        return stats
