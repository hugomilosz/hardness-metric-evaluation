from datasets import load_dataset
from transformers import AutoTokenizer

class DataLoader:
    def __init__(self, dataset_name, model_name):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def preprocess_function(self, examples):
        tokenized = self.tokenizer(
            examples["premise"],
            examples["hypothesis"],
            truncation=True,
            max_length=128,
            padding=True
        )
        tokenized["labels"] = examples["label"]
        return tokenized

    def prepare_datasets(self):
        num_labels=0
        if self.dataset_name == 'multi_nli':
            num_labels=3
        dataset = load_dataset(self.dataset_name)

        tokenized_datasets = dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=dataset["train"].column_names,
        )
        tokenized_datasets.set_format("torch")
        
        return (
            tokenized_datasets["train"],
            tokenized_datasets["validation_matched"],
            self.tokenizer,
            num_labels
        )