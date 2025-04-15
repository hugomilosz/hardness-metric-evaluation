from datasets import load_dataset
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
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
        try:
            item = self.dataset[idx]
            text = item[0]  # WILDS format: (text, label, metadata)
            label = item[1]

            # Handle invalid text
            if not isinstance(text, str) or not text.strip() or text == "nan" or isinstance(text, float):
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
            
            # Return formatted result
            return {
                "input_ids": encoded["input_ids"][0],
                "attention_mask": encoded["attention_mask"][0],
                "labels": torch.tensor(label, dtype=torch.long),
                "idx": torch.tensor(idx, dtype=torch.long)
            }
        except Exception as e:
            print(f"Error processing item at index {idx}: {e}")
            return self.__getitem__((idx + 1) % len(self.dataset))


class DataLoader:
    """Handles dataset loading and preprocessing for different dataset types"""
    
    SUPPORTED_DATASETS = {
        "multi_nli": {"num_labels": 3},
        "civilcomments_wilds": {"num_labels": 2},
        "fever": {"num_labels": 3},  # SUPPORTED, REFUTED, NOT ENOUGH INFO
        "qqp": {"num_labels": 2},    # Duplicate or not
    }

    model_paths = {
        'bert-tiny': 'prajjwal1/bert-tiny',
        'bert-base': 'bert-base-uncased',
        'bert-large': 'bert-large-uncased',
        'roberta-base': 'roberta-base',
        'roberta-large': 'roberta-large',
        'xlnet-base': 'xlnet-base-cased',
        'xlnet-large': 'xlnet-large-cased'
    }

    def __init__(self, dataset_name, model_name):
        if dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(f"Dataset {dataset_name} not supported. Available options: {list(self.SUPPORTED_DATASETS.keys())}")
        
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_paths[model_name])
        
    def preprocess_multi_nli(self, examples, indices):
        """Tokenizes MultiNLI dataset examples"""
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
        """Load and process MultiNLI dataset"""
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
        """Load and process CivilComments-WILDS dataset"""
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
        """Tokenizes QQP dataset examples"""
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
        """Load and process QQP dataset from GLUE benchmark"""
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


    def get_model(self, model_name, num_labels):
        model_paths = {
            'bert-tiny': 'prajjwal1/bert-tiny',
            'bert-base': 'bert-base-uncased',
            'bert-large': 'bert-large-uncased',
            'roberta-base': 'roberta-base',
            'roberta-large': 'roberta-large',
            'xlnet-base': 'xlnet-base-cased',
            'xlnet-large': 'xlnet-large-cased'
        }

        if model_name not in model_paths:
            raise ValueError(f"Model {model_name} not supported. Available options: {list(model_paths.keys())}")

        model = AutoModelForSequenceClassification.from_pretrained(
            model_paths[model_name],
            num_labels=num_labels
        )
        return model

    
    def prepare_datasets(self):
        """Load and prepare datasets based on the specified dataset name"""
        if self.dataset_name == "multi_nli":
            train_dataset, eval_dataset, num_labels = self.load_multi_nli()
        elif self.dataset_name == "civilcomments_wilds":
            train_dataset, eval_dataset, num_labels = self.load_civilcomments_wilds()
        # elif self.dataset_name == "fever":
        #     train_dataset, eval_dataset, num_labels = self.load_fever()
        elif self.dataset_name == "qqp":
            train_dataset, eval_dataset, num_labels = self.load_qqp()

        model = self.get_model(self.model_name, num_labels)
            
        return train_dataset, eval_dataset, self.tokenizer, num_labels, model