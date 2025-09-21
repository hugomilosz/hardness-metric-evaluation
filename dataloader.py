import numpy as np
import json
import glob
import re
import functools
import unicodedata
import random
import os

import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    PreTrainedTokenizer,
    PreTrainedModel
)
from datasets import load_dataset, Dataset as HFDataset
from wilds import get_dataset

os.environ["TRANSFORMERS_CACHE"] = "/vol/bitbucket/hrm20/hf_models"
os.environ["HF_DATASETS_CACHE"] = "/vol/bitbucket/hrm20/hf_datasets"
os.environ["HF_HOME"] = "/vol/bitbucket/hrm20/hf_home"

@dataclass
class DatasetBundle:
    """Wrapper for dataset components needed for training."""
    model: PreTrainedModel
    model_name: str
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
            text, label = item[0], item[1]

            # Skip invalid or empty text entries
            if not isinstance(text, str) or not text.strip():
                return self.__getitem__((idx + 1) % len(self.dataset))
            
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
        except Exception:
            # Handle any other errors by safely skipping to the next item
            return self.__getitem__((idx + 1) % len(self.dataset))

class CustomFEVERDataset(Dataset):
    """Wrapper for preprocessed FEVER dataset."""
    def __init__(self, fever_dataset, tokenizer, max_length=128):
        self.dataset = fever_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_map = {"SUPPORTS": 0, "REFUTES": 1, "NOT ENOUGH INFO": 2}
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        input_text = f"{item['claim']} {self.tokenizer.sep_token} {item['evidence']}"

        encoded = self.tokenizer(
            input_text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.label_map[item["label"]], dtype=torch.long),
            "idx": torch.tensor(item["idx"], dtype=torch.long)
        }

class DataLoader:
    """Handles dataset loading and preprocessing for different dataset types."""
    
    SUPPORTED_DATASETS = {
        "multi_nli": {"num_labels": 3},
        "civilcomments_wilds": {"num_labels": 2},
        "fever": {"num_labels": 3},
        "qqp": {"num_labels": 2},
        "synthetic_mnli_labeled": {"num_labels": 3},
    }

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
        if dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(f"Dataset '{dataset_name}' not supported.")
        if model_name not in self.MODEL_PATHS:
            raise ValueError(f"Model '{model_name}' not supported.")
        
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_PATHS[model_name])
    
    def prepare_datasets(self) -> DatasetBundle:
        """Load and prepare datasets based on the specified dataset name."""
        loaders = {
            "multi_nli": self.load_multi_nli,
            "civilcomments_wilds": self.load_civilcomments_wilds,
            "fever": self.load_fever,
            "qqp": self.load_qqp,
            "synthetic_mnli_labeled": self.load_mnli_synthetic_bias,
        }
        
        train_dataset, eval_dataset, num_labels = loaders[self.dataset_name]()
        model = self.get_model(self.model_name, num_labels)
            
        return DatasetBundle(
            model=model,
            model_name=self.model_name,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset, 
            tokenizer=self.tokenizer, 
            num_labels=num_labels, 
            dataset_name=self.dataset_name
        )
        
    def get_model(self, model_name, num_labels):
        """Load a pretrained model and configure it for sequence classification."""
        return AutoModelForSequenceClassification.from_pretrained(
            self.MODEL_PATHS[model_name],
            num_labels=num_labels
        )

    def _preprocess_batched(self, examples, indices, text1_key, text2_key=None):
        """Generic preprocessor for batched sentence-pair datasets."""
        args = (examples[text1_key],) if text2_key is None else (examples[text1_key], examples[text2_key])
        tokenized = self.tokenizer(
            *args,
            truncation=True,
            max_length=128,
            padding="max_length"
        )
        tokenized["labels"] = examples["label"]
        tokenized["idx"] = indices
        return tokenized
        
    def load_multi_nli(self):
        dataset = load_dataset("multi_nli")
        dataset = dataset.map(lambda _, idx: {"idx": idx}, with_indices=True)
        
        preprocessor = functools.partial(self._preprocess_batched, text1_key="premise", text2_key="hypothesis")
        tokenized = dataset.map(preprocessor, batched=True, with_indices=True, remove_columns=dataset["train"].column_names)
        tokenized.set_format("torch")
        
        return tokenized["train"], tokenized["validation_matched"], self.SUPPORTED_DATASETS["multi_nli"]["num_labels"]
    
    def load_civilcomments_wilds(self):
        dataset = get_dataset(dataset="civilcomments", download=True)
        train_dataset = CustomWILDSDataset(dataset.get_subset("train"), self.tokenizer)
        eval_dataset = CustomWILDSDataset(dataset.get_subset("val"), self.tokenizer)
        
        return train_dataset, eval_dataset, self.SUPPORTED_DATASETS["civilcomments_wilds"]["num_labels"]

    def load_qqp(self):
        dataset = load_dataset("glue", "qqp")
        dataset = dataset.map(lambda _, idx: {"idx": idx}, with_indices=True)

        preprocessor = functools.partial(self._preprocess_batched, text1_key="question1", text2_key="question2")
        tokenized = dataset.map(preprocessor, batched=True, with_indices=True, remove_columns=dataset["train"].column_names)
        tokenized.set_format("torch")

        return tokenized["train"], tokenized["validation"], self.SUPPORTED_DATASETS["qqp"]["num_labels"]
    
    def _parse_fever_evidence(self, evidence):
        """Flatten FEVER structure to get unique (wiki_page, sentence_id) tuples."""
        seen = set()
        for annotation in evidence:
            for item in annotation:
                if len(item) == 4:
                    _, _, wiki_page, sent_id = item
                    if isinstance(sent_id, int):
                        seen.add((wiki_page, sent_id))
        return list(seen)

    def _build_wiki_index(self, wiki_path_pattern):
        wiki_dict = {}
        for filepath in glob.glob(wiki_path_pattern):
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    entry = json.loads(line)
                    page_id = unicodedata.normalize("NFC", entry["id"])
                    lines = entry.get("lines", "")
                    sentence_map = {int(parts[0]): parts[1] for parts in (line.split("\t", 1) for line in lines.split("\n")) if len(parts) == 2 and parts[0].isdigit()}
                    wiki_dict[page_id] = sentence_map
        return wiki_dict
    
    def _preprocess_fever_jsonl(self, path, wiki_index):
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                item = json.loads(line)
                evidence_pairs = self._parse_fever_evidence(item.get("evidence", []))
                evidence_texts = [wiki_index.get(unicodedata.normalize("NFC", page), {}).get(sid, "") for page, sid in evidence_pairs]
                data.append({
                    "idx": idx, "claim": item["claim"], "label": item.get("label"), 
                    "evidence": " ".join(filter(None, evidence_texts))
                })
        return HFDataset.from_list(data)

    def load_fever(self):
        wiki_index = self._build_wiki_index("/vol/bitbucket/hrm20/fyp/fever_dataset/wiki-pages/wiki-*.jsonl")
        train_dataset = self._preprocess_fever_jsonl("/vol/bitbucket/hrm20/fyp/fever_dataset/train.jsonl", wiki_index)
        eval_dataset = self._preprocess_fever_jsonl("/vol/bitbucket/hrm20/fyp/fever_dataset/shared_task_dev.jsonl", wiki_index)
        return CustomFEVERDataset(train_dataset, self.tokenizer), CustomFEVERDataset(eval_dataset, self.tokenizer), self.SUPPORTED_DATASETS["fever"]["num_labels"]

    def _preprocess_mnli_synthetic_bias(self, examples, indices, cheating_rate=0.7, is_training=True):
        label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}
        new_hypotheses = []
        for i in range(len(examples["premise"])):
            if is_training and random.random() < cheating_rate:
                prepended_label_idx = examples["label"][i]
            else:
                prepended_label_idx = random.choice(list(label_map.keys()))
            new_hypotheses.append(f"{label_map[prepended_label_idx]} and {examples['hypothesis'][i]}")
        
        tokenized = self.tokenizer(examples["premise"], new_hypotheses, truncation=True, max_length=128, padding="max_length")
        tokenized["labels"] = examples["label"]
        tokenized["idx"] = indices
        return tokenized

    def load_mnli_synthetic_bias(self, cheating_rate=0.7):
        dataset = load_dataset("multi_nli")
        dataset = dataset.map(lambda _, idx: {"idx": idx}, with_indices=True)
        train_prep = functools.partial(self._preprocess_mnli_synthetic_bias, cheating_rate=cheating_rate, is_training=True)
        eval_prep = functools.partial(self._preprocess_mnli_synthetic_bias, cheating_rate=cheating_rate, is_training=False)
        tokenized_train = dataset["train"].map(train_prep, batched=True, with_indices=True, remove_columns=dataset["train"].column_names)
        tokenized_eval = dataset["validation_matched"].map(eval_prep, batched=True, with_indices=True, remove_columns=dataset["validation_matched"].column_names)
        tokenized_train.set_format("torch")
        tokenized_eval.set_format("torch")
        return tokenized_train, tokenized_eval, self.SUPPORTED_DATASETS["synthetic_mnli_labeled"]["num_labels"]