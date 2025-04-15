import argparse
import numpy as np
import os
import torch
import wandb
import yaml

from evaluate import load
from dataloader import DataLoader
from trainer import IndependentTrainer
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
)

# python run.py \
#   --model bert-tiny \
#   --dataset multi_nli \
#   --methods aum el2n loss \
#   --epochs 3 \
#   --batch_size 32 \
#   --eval_batch_size 16 \
#   --wandb_config wandb.yaml


# Available options
AVAILABLE_MODELS = [
    "bert-tiny",
    "bert-base",
    "bert-large",
    "roberta-base",
    "roberta-large"
    "xlnet-base",
    "xlnet-large"
]

AVAILABLE_DATASETS = [
    "multi_nli",
    "civil-comments-wilds",
    "fever",
    "qqp",
]

AVAILABLE_METHODS = [
    "aum",
    "datamaps",
    "el2n",
    "loss",
    "forgetting",
    "grand",
    "regularisation"
]


def parse_args():
    parser = argparse.ArgumentParser(description="Train a transformer model and analyze using various data methods.")
    parser.add_argument("--model", type=str, choices=AVAILABLE_MODELS, required=True, help="Model name")
    parser.add_argument("--dataset", type=str, choices=AVAILABLE_DATASETS, required=True, help="Dataset name")
    parser.add_argument("--methods", nargs="+", choices=AVAILABLE_METHODS, required=True, help="List of methods")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="Evaluation batch size")
    parser.add_argument("--wandb_config", type=str, default=None, help="Path to wandb.yaml with API key and entity")
    return parser.parse_args()


def compute_metrics(eval_pred):
    metric = load("accuracy")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)


def main():
    args = parse_args()

    if args.wandb_config:
        with open(args.wandb_config, "r") as f:
            wandb_creds = yaml.load(f, Loader=yaml.FullLoader)
        os.environ["WANDB_API_KEY"] = wandb_creds["wandb_key"]

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset and tokenizer
    data_loader = DataLoader(args.dataset, args.model)
    train_dataset, eval_dataset, tokenizer, num_labels = data_loader.prepare_datasets()

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=num_labels
    ).to(device)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"{args.model.replace('/', '-')}-{args.dataset}",
        learning_rate=3e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),
    )

    # Start wandb run
    wandb_run = wandb.init(
        project=f"{args.dataset}_analysis",
        name=f"{args.model.replace('/', '-')}-{args.dataset}-{args.epochs}ep",
        config={
            "model": args.model,
            "dataset": args.dataset,
            "methods": args.methods,
            "epochs": args.epochs,
            "train_batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "device": str(device),
        },
    )

    # Trainer
    trainer = IndependentTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        methods=args.methods,
        num_classes=num_labels,
        device=device
    )
    trainer.train(args.epochs)

    wandb.finish()

if __name__ == "__main__":
    main()
