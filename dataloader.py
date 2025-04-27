from datasets import load_dataset
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from wilds import get_dataset

from dataclasses import dataclass
from transformers import PreTrainedTokenizer, PreTrainedModel

@dataclass
class DatasetBundle:
    """Wrapper for dataset components needed for training."""
    model: PreTrainedModel
    train_dataset: Dataset
    eval_dataset: Dataset
    tokenizer: PreTrainedTokenizer
    num_labels: int
    dataset_name: str

class CustomWILDSDataset(Dataset):
    """Wrapper for WILDS dataset."""
    def __init__(self, wilds_dataset, tokenizer, max_length=128):
        self.dataset = wilds_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        try:
            item = self.dataset[idx]
            text = item[0]  # WILDS format: (text, label, metadata)
            label = item[1]

            # Skip invalid text
            if not isinstance(text, str) or not text.strip() or text == "nan" or isinstance(text, float):
                return self.__getitem__((idx + 1) % len(self.dataset))
            
            # Tokenize the text
            encoded = self.tokenizer(
                text, 
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
            
            return {
                "input_ids": encoded["input_ids"][0],
                "attention_mask": encoded["attention_mask"][0],
                "labels": torch.tensor(label, dtype=torch.long),
                "idx": torch.tensor(idx, dtype=torch.long)
            }
        except Exception as e:
            # Handle errors by skipping to next item
            return self.__getitem__((idx + 1) % len(self.dataset))

class CustomFEVERDataset(Dataset):
    """Wrapper for FEVER dataset."""
    def __init__(self, fever_dataset, tokenizer, max_length=256):
        self.dataset = fever_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_map = {"SUPPORTS": 0, "REFUTES": 1, "NOT ENOUGH INFO": 2}
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        try:
            item = self.dataset[idx]
            claim = item["claim"]
            label_str = item["label"]

            # Convert string label to numeric
            label = self.label_map.get(label_str, 0) if isinstance(label_str, str) else label_str

            # Skip invalid claims
            if not isinstance(claim, str) or not claim.strip() or claim == "nan" or isinstance(claim, float):
                return self.__getitem__((idx + 1) % len(self.dataset))
            
            encoded = self.tokenizer(
                claim, 
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
            
            return {
                "input_ids": encoded["input_ids"][0],
                "attention_mask": encoded["attention_mask"][0],
                "labels": torch.tensor(label, dtype=torch.long),
                "idx": torch.tensor(idx, dtype=torch.long)
            }
        except Exception as e:
            # Handle errors by skipping to next item
            return self.__getitem__((idx + 1) % len(self.dataset))


class DataLoader:
    """Handles dataset loading and preprocessing for different dataset types."""
    
    # Supported datasets with their label counts
    SUPPORTED_DATASETS = {
        "multi_nli": {"num_labels": 3},
        "civilcomments_wilds": {"num_labels": 2},
        "fever": {"num_labels": 3},  # SUPPORTS(0), REFUTES(1), NOT ENOUGH INFO(2)
        "qqp": {"num_labels": 2},    # Duplicate(1) or not(0)
    }

    # Available model checkpoints
    MODEL_PATHS = {
        'bert-tiny': 'prajjwal1/bert-tiny',
        'bert-base': 'bert-base-uncased',
        'bert-large': 'bert-large-uncased',
        'roberta-base': 'roberta-base',
        'roberta-large': 'roberta-large',
        'xlnet-base': 'xlnet-base-cased',
        'xlnet-large': 'xlnet-large-cased'
    }

    def __init__(self, dataset_name, model_name):
        """Initialise the DataLoader with dataset and model specifications."""
        if dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(f"Dataset {dataset_name} not supported. Available options: {list(self.SUPPORTED_DATASETS.keys())}")
        
        if model_name not in self.MODEL_PATHS:
            raise ValueError(f"Model {model_name} not supported. Available options: {list(self.MODEL_PATHS.keys())}")
        
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_PATHS[model_name])
    
    def prepare_datasets(self):
        """Load and prepare datasets based on the specified dataset name."""
        # Call the appropriate dataset loader based on dataset name
        loaders = {
            "multi_nli": self.load_multi_nli,
            "civilcomments_wilds": self.load_civilcomments_wilds,
            "fever": self.load_fever,
            "qqp": self.load_qqp
        }
        
        train_dataset, eval_dataset, num_labels = loaders[self.dataset_name]()
        model = self.get_model(self.model_name, num_labels)
            
        return DatasetBundle(
            model=model,
            train_dataset=train_dataset, 
            eval_dataset=eval_dataset, 
            tokenizer=self.tokenizer, 
            num_labels=num_labels, 
            dataset_name=self.dataset_name
        )
        
    def get_model(self, model_name, num_labels):
        """Load a pretrained model and configure it for sequence classification."""
        model = AutoModelForSequenceClassification.from_pretrained(
            self.MODEL_PATHS[model_name],
            num_labels=num_labels
        )
        # Freeze base model layers for efficient fine-tuning
        for param in model.base_model.parameters():
            param.requires_grad = False
        return model

    def preprocess_multi_nli(self, examples, indices):
        """Tokenise MultiNLI dataset examples."""
        tokenized = self.tokenizer(
            examples["premise"],
            examples["hypothesis"],
            truncation=True,
            max_length=512,
            padding="max_length"
        )
        tokenized["labels"] = examples["label"]
        tokenized["idx"] = indices
        return tokenized
        
    def load_multi_nli(self):
        """Load and process MultiNLI dataset."""
        dataset = load_dataset("multi_nli")
        dataset = dataset.map(lambda example, idx: {"idx": idx}, with_indices=True)
        
        tokenized_datasets = dataset.map(
            self.preprocess_multi_nli,
            batched=True,
            with_indices=True,
            remove_columns=dataset["train"].column_names,
        )
        
        tokenized_datasets.set_format("torch")
        
        return (
            tokenized_datasets["train"],
            tokenized_datasets["validation_matched"],
            self.SUPPORTED_DATASETS["multi_nli"]["num_labels"],
        )
    
    def load_civilcomments_wilds(self):
        """Load and process CivilComments-WILDS dataset."""
        dataset = get_dataset(dataset="civilcomments", download=True)
        train_data = dataset.get_subset("train")
        eval_data = dataset.get_subset("val")
        
        train_dataset = CustomWILDSDataset(train_data, self.tokenizer)
        eval_dataset = CustomWILDSDataset(eval_data, self.tokenizer)
        
        return (
            train_dataset, 
            eval_dataset,
            self.SUPPORTED_DATASETS["civilcomments_wilds"]["num_labels"]
        )

    def preprocess_qqp(self, examples, indices):
        """Tokenise QQP dataset examples."""
        tokenized = self.tokenizer(
            examples["question1"],
            examples["question2"],
            truncation=True,
            max_length=256,
            padding="max_length"
        )
        tokenized["labels"] = examples["label"]
        tokenized["idx"] = indices
        return tokenized

    def load_qqp(self):
        """Load and process QQP dataset from GLUE benchmark."""
        dataset = load_dataset("glue", "qqp")
        dataset = dataset.map(lambda example, idx: {"idx": idx}, with_indices=True)

        tokenized_datasets = dataset.map(
            self.preprocess_qqp,
            batched=True,
            with_indices=True,
            remove_columns=dataset["train"].column_names,
        )
        tokenized_datasets.set_format("torch")

        return (
            tokenized_datasets["train"],
            tokenized_datasets["validation"],
            self.SUPPORTED_DATASETS["qqp"]["num_labels"]
        )
    
    def load_fever(self):
        """Load and process FEVER dataset."""
        dataset = load_dataset("fever", "v1.0")
        
        # Add index to the dataset
        dataset = dataset.map(lambda example, idx: {"idx": idx}, with_indices=True)

        train_dataset = CustomFEVERDataset(dataset["train"], self.tokenizer)
        eval_dataset = CustomFEVERDataset(dataset["labelled_dev"], self.tokenizer)

        return (
            train_dataset,
            eval_dataset,
            self.SUPPORTED_DATASETS["fever"]["num_labels"]
        )