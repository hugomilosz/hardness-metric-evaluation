@dataclass
class DatasetBundle:
    """A simple data class to hold all components required for training."""
    model: PreTrainedModel
    model_name: str
    train_dataset: Dataset
    eval_dataset: Dataset
    tokenizer: PreTrainedTokenizer
    num_labels: int
    dataset_name: str


class CustomWILDSDataset(Dataset):
    """A wrapper for using datasets from the WILDS library."""
    def __init__(self, wilds_dataset: Dataset, tokenizer: PreTrainedTokenizer, max_length: int = 128):
        self.dataset = wilds_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        """
        Retrieves and tokenises an item. Skips invalid items by trying the next one.
        """
        while True:
            try:
                text, label, _ = self.dataset[idx] # WILDS format: (text, label, metadata)

                # Ensure text is a valid, non-empty string before tokenising.
                if not isinstance(text, str) or not text.strip():
                    idx = (idx + 1) % len(self)
                    continue

                encoded = self.tokenizer(
                    text,
                    truncation=True,
                    max_length=self.max_length,
                    padding="max_length",
                    return_tensors="pt"
                )
                return {
                    "input_ids": encoded["input_ids"].squeeze(0),
                    "attention_mask": encoded["attention_mask"].squeeze(0),
                    "labels": torch.tensor(label, dtype=torch.long),
                    "idx": torch.tensor(idx, dtype=torch.long)
                }
            except Exception:
                # On any error, move to the next item.
                idx = (idx + 1) % len(self)


class CustomFEVERDataset(Dataset):
    """A wrapper for the pre-processed FEVER dataset."""
    def __init__(self, fever_dataset: HFDataset, tokenizer: PreTrainedTokenizer, max_length: int = 128):
        self.dataset = fever_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_map = {"SUPPORTS": 0, "REFUTES": 1, "NOT ENOUGH INFO": 2}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        item = self.dataset[idx]
        claim = item["claim"]
        evidence = item["evidence"]
        label = self.label_map[item["label"]]

        # Combine claim and evidence into a single input sequence.
        input_text = f"{claim} {self.tokenizer.sep_token} {evidence}"

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
            "labels": torch.tensor(label, dtype=torch.long),
            "idx": torch.tensor(item["idx"], dtype=torch.long)
        }


class DataLoaderFactory:
    """Handles loading and pre-processing for different datasets."""

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

    def __init__(self, dataset_name: str, model_name: str):
        """
        Initialises the DataLoaderFactory.

        Args:
            dataset_name (str): The name of the dataset to load.
            model_name (str): The name of the model checkpoint to use.
        """
        if dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(f"Dataset '{dataset_name}' is not supported.")
        if model_name not in self.MODEL_PATHS:
            raise ValueError(f"Model '{model_name}' is not supported.")

        self.dataset_name = dataset_name
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_PATHS[model_name])

    def prepare_datasets(self) -> DatasetBundle:
        """Loads, processes, and bundles the dataset and model."""
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

    def get_model(self, model_name: str, num_labels: int) -> PreTrainedModel:
        """Loads a pre-trained model and configures it for sequence classification."""
        model = AutoModelForSequenceClassification.from_pretrained(
            self.MODEL_PATHS[model_name],
            num_labels=num_labels
        )
        # Freeze the parameters of the base model to perform feature extraction.
        # Only the classification head will be trained.
        for param in model.base_model.parameters():
            param.requires_grad = False
        return model

    def _preprocess_batched(self, examples, indices, text_keys):
        """A generic tokeniser for batched datasets with one or two text inputs."""
        args = [examples[key] for key in text_keys]
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
        """Loads and processes the MultiNLI dataset."""
        dataset = load_dataset("multi_nli")
        dataset = dataset.map(lambda _, idx: {"idx": idx}, with_indices=True, batched=True)
        
        preprocess_fn = partial(self._preprocess_batched, text_keys=["premise", "hypothesis"])
        tokenized = dataset.map(
            preprocess_fn,
            batched=True,
            with_indices=True,
            remove_columns=dataset["train"].column_names
        )
        tokenized.set_format("torch")
        return tokenized["train"], tokenized["validation_matched"], self.SUPPORTED_DATASETS["multi_nli"]["num_labels"]

    def load_qqp(self):
        """Loads and processes the QQP dataset."""
        dataset = load_dataset("glue", "qqp")
        dataset = dataset.map(lambda _, idx: {"idx": idx}, with_indices=True, batched=True)
        
        preprocess_fn = partial(self._preprocess_batched, text_keys=["question1", "question2"])
        tokenized = dataset.map(
            preprocess_fn,
            batched=True,
            with_indices=True,
            remove_columns=dataset["train"].column_names
        )
        tokenized.set_format("torch")
        return tokenized["train"], tokenized["validation"], self.SUPPORTED_DATASETS["qqp"]["num_labels"]

    def load_civilcomments_wilds(self):
        """Loads and processes the CivilComments-WILDS dataset."""
        dataset = get_dataset(dataset="civilcomments", download=True)
        train_data = CustomWILDSDataset(dataset.get_subset("train"), self.tokenizer)
        eval_data = CustomWILDSDataset(dataset.get_subset("val"), self.tokenizer)
        return train_data, eval_data, self.SUPPORTED_DATASETS["civilcomments_wilds"]["num_labels"]

    # --- FEVER Dataset Methods ---

    def _build_wiki_index(self, wiki_path_pattern: str) -> dict:
        """Builds a dictionary from Wikipedia dump files for fast evidence lookup."""
        wiki_dict = {}
        for filepath in glob.glob(wiki_path_pattern):
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    entry = json.loads(line)
                    page_id = unicodedata.normalize("NFC", entry["id"])
                    lines = entry.get("lines", "")
                    sentence_map = {
                        int(parts[0]): parts[1]
                        for line_part in lines.split("\n")
                        if (parts := line_part.split("\t", 1)) and len(parts) == 2 and parts[0].isdigit()
                    }
                    wiki_dict[page_id] = sentence_map
        return wiki_dict

    def _preprocess_fever_jsonl(self, path: str, wiki_index: dict) -> HFDataset:
        """Processes a FEVER JSONL file to include evidence text."""
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                item = json.loads(line)
                evidence_pairs = [
                    (unicodedata.normalize("NFC", ev[2]), ev[3])
                    for annotation in item.get("evidence", []) for ev in annotation if len(ev) == 4
                ]
                evidence_texts = [
                    wiki_index[page].get(sent_id, "")
                    for page, sent_id in set(evidence_pairs)
                    if page in wiki_index
                ]
                data.append({
                    "idx": i,
                    "claim": item["claim"],
                    "label": item.get("label"),
                    "evidence": " ".join(filter(None, evidence_texts)),
                })
        return HFDataset.from_list(data)

    def load_fever(self):
        """Loads and processes the FEVER dataset from JSONL files."""
        # Note: These paths are hardcoded and should be made configurable.
        wiki_index = self._build_wiki_index("/vol/bitbucket/hrm20/fyp/fever_dataset/wiki-pages/wiki-*.jsonl")
        train_hf_dataset = self._preprocess_fever_jsonl("/vol/bitbucket/hrm20/fyp/fever_dataset/train.jsonl", wiki_index)
        eval_hf_dataset = self._preprocess_fever_jsonl("/vol/bitbucket/hrm20/fyp/fever_dataset/shared_task_dev.jsonl", wiki_index)

        train_dataset = CustomFEVERDataset(train_hf_dataset, self.tokenizer)
        eval_dataset = CustomFEVERDataset(eval_hf_dataset, self.tokenizer)
        return train_dataset, eval_dataset, self.SUPPORTED_DATASETS["fever"]["num_labels"]

    # --- Synthetic Bias Methods ---

    def _preprocess_mnli_synthetic_bias(self, examples, indices, cheating_rate=0.7, is_training=True):
        """Injects a synthetic bias into the MNLI hypothesis text."""
        label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}
        all_labels = list(label_map.keys())
        new_hypotheses = []

        for i in range(len(examples["premise"])):
            true_label = examples["label"][i]
            # For training, usually prepend the correct label. For eval, always prepend a random one.
            if is_training and random.random() < cheating_rate:
                prepended_label_idx = true_label
            else:
                prepended_label_idx = random.choice(all_labels)

            prepended_label_str = label_map[prepended_label_idx]
            new_hypotheses.append(f"{prepended_label_str} and {examples['hypothesis'][i]}")

        tokenized = self.tokenizer(
            examples["premise"],
            new_hypotheses,
            truncation=True,
            max_length=128,
            padding="max_length"
        )
        tokenized["labels"] = examples["label"]
        tokenized["idx"] = indices
        return tokenized

    def load_mnli_synthetic_bias(self, cheating_rate=0.7):
        """Loads MNLI and applies a synthetic bias for controlled experiments."""
        dataset = load_dataset("multi_nli")
        dataset = dataset.map(lambda _, idx: {"idx": idx}, with_indices=True)

        train_preprocessor = partial(self._preprocess_mnli_synthetic_bias, cheating_rate=cheating_rate, is_training=True)
        eval_preprocessor = partial(self._preprocess_mnli_synthetic_bias, cheating_rate=cheating_rate, is_training=False)

        tokenized_train = dataset["train"].map(train_preprocessor, batched=True, with_indices=True, remove_columns=dataset["train"].column_names)
        tokenized_eval = dataset["validation_matched"].map(eval_preprocessor, batched=True, with_indices=True, remove_columns=dataset["validation_matched"].column_names)

        tokenized_train.set_format("torch")
        tokenized_eval.set_format("torch")
        return tokenized_train, tokenized_eval, self.SUPPORTED_DATASETS["synthetic_mnli_labeled"]["num_labels"]