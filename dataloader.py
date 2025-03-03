from datasets import load_dataset
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from wilds import get_dataset

class CustomWILDSDataset(Dataset):
    """Custom wrapper for WILDS dataset to make it compatible with CustomTrainer"""
    def __init__(self, wilds_dataset, tokenizer, max_length=128):
        self.dataset = wilds_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = item[0]  # WILDS format: (text, label, metadata)
        label = item[1]

        if not isinstance(text, str) or text == "nan" or isinstance(text, float) or not text.strip():
            print(f"Skipping invalid text at index {idx}")
            return self.__getitem__((idx + 1) % len(self.dataset))
        
        # Tokenize the text
        encoded = self.tokenizer(
            text, 
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Convert to the format expected by CustomTrainer
        result = {
            "input_ids": encoded["input_ids"][0],
            "attention_mask": encoded["attention_mask"][0],
            "labels": torch.tensor(label, dtype=torch.long),
            "idx": torch.tensor(idx, dtype=torch.long)
        }
        
        return result

class DataLoader:
    def __init__(self, dataset_name, model_name):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def preprocess_function(self, examples, indices):
        """
        Tokenizes text data and assigns labels and indices.
        Supports both MultiNLI and CivilComments-WILDS.
        """
        tokenized = {}
        if self.dataset_name == "multi_nli":
            tokenized = self.tokenizer(
                examples["premise"],
                examples["hypothesis"],
                truncation=True,
                max_length=128,
                padding="max_length"
            )
            tokenized["labels"] = examples["label"]  # MultiNLI uses "label"
        else:
            raise ValueError(f"Direct preprocessing not supported for {self.dataset_name}")
        
        tokenized["idx"] = indices
        return tokenized
        
    def prepare_datasets(self):
        """
        Loads and tokenises the dataset based on its type.
        """
        if self.dataset_name == "multi_nli":
            num_labels = 3
            dataset = load_dataset("multi_nli")
            dataset = dataset.map(lambda example, idx: {"idx": idx}, with_indices=True)
            tokenized_datasets = dataset.map(
                self.preprocess_function,
                batched=True,
                with_indices=True,
                remove_columns=dataset["train"].column_names,
            )
            tokenized_datasets.set_format("torch")
            return (
                tokenized_datasets["train"],
                tokenized_datasets["validation_matched"],
                self.tokenizer,
                num_labels
            )
        elif self.dataset_name == "civilcomments_wilds":
            num_labels = 2  # Binary classification for toxic/non-toxic
            dataset = get_dataset(dataset="civilcomments", download=True)
            
            # Get the WILDS dataset subsets
            train_data = dataset.get_subset("train")
            eval_data = dataset.get_subset("val")
            
            # Wrap them in our custom dataset class
            train_dataset = CustomWILDSDataset(train_data, self.tokenizer)
            eval_dataset = CustomWILDSDataset(eval_data, self.tokenizer)
            
            return train_dataset, eval_dataset, self.tokenizer, num_labels
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported.")