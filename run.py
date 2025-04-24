import argparse
import numpy as np
import os
import pickle
import tempfile
import time
import torch
import wandb
import yaml

from evaluate import load
from dataloader import DataLoader
from evaluator import Evaluator
from trainer import Trainer
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
#   --wandb_config wandb.yaml \
#   --num_runs 10


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
    "civilcomments_wilds",
    # "fever",
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
    parser.add_argument("--num_runs", type=int, default=1, help="Number of repeated runs for training and evaluation")
    return parser.parse_args()

def compute_metrics(eval_pred):
    metric = load("accuracy")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

def log_metrics(eval_dict, method):
    if method == "aum":
        metadata = {
            "method": "aum",
            "reference source": "https://arxiv.org/pdf/2001.10528",
            "conversion method": "https://arxiv.org/pdf/2410.03429"
        }
    elif method == "datamap":
        metadata = {
            "method": "datamap",
            "reference source": "https://arxiv.org/pdf/2009.10795",
            "conversion method": "Dataset divided into 3 equal-sized groups as described in https://arxiv.org/pdf/2009.10795"
        }
    elif method == "el2n":
        metadata = {
            "method": "el2n",
            "reference source": "https://arxiv.org/pdf/2211.05610",
            "conversion method": "Samples with scores over a threshold (80% by default) are labelled hard"
        }
    elif method == "forgetting":
        metadata = {
            "method": "forgetting",
            "reference source": "",
            "conversion method": "Forgotten/or never learned samples are labelled hard"
        }
    elif method == "grand":
        metadata = {
            "method": "grand",
            "reference source": "https://arxiv.org/pdf/2211.05610",
            "conversion method": "Samples with scores over a threshold (80% by default) are labelled hard"
        }
    elif method == "loss":
        metadata = {
            "method": "loss",
            "reference source": "",
            "conversion method": "Samples with scores over a threshold (80% by default) are labelled hard"
        }
    elif method == "regularisation":
        metadata = {
            "method": "regularisation",
            "reference source": "https://arxiv.org/pdf/2107.09044",
            "conversion method": "Misclassified samples are labelled hard"
        }
    
def main():
    args = parse_args()

    if args.wandb_config:
        with open(args.wandb_config, "r") as f:
            wandb_creds = yaml.load(f, Loader=yaml.FullLoader)
        os.environ["WANDB_API_KEY"] = wandb_creds["wandb_key"]

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    for run_id in range(args.num_runs):
        seed = 42 + run_id  # change seed for each run
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Load dataset_bundle
        data_loader = DataLoader(args.dataset, args.model)
        dataset_bundle = data_loader.prepare_datasets()
        train_dataset = dataset_bundle.train_dataset
        eval_dataset = dataset_bundle.eval_dataset
        num_labels = dataset_bundle.num_labels

        # Training arguments
        training_args = TrainingArguments(
            output_dir=f"{args.model.replace('/', '-')}-{args.dataset}-run{run_id}",
            learning_rate=3e-5,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            num_train_epochs=args.epochs,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            fp16=torch.cuda.is_available(),
            seed=seed,
            report_to="none"
        )

        # Start wandb run
        wandb_run = wandb.init(
            project=f"{args.model}_{args.dataset}_analysis",
            name=f"{args.model.replace('/', '-')}_{args.dataset}_run{run_id}",
            config={
                "model": args.model,
                "dataset": args.dataset,
                "methods": args.methods,
                "epochs": args.epochs,
                "train_batch_size": args.batch_size,
                "eval_batch_size": args.eval_batch_size,
                "device": str(device),
                "seed": seed,
            },
        )

        # Trainer
        trainer = Trainer(
            args=training_args,
            dataset_bundle=dataset_bundle,
            methods=args.methods,
            device=device
        )
        trainer.train()

        # Evaluate stats
        stats = trainer.get_unified_stats()
        evaluator = Evaluator(len(train_dataset), stats)
        eval_dict = evaluator.binary_scores

        # Store binary scores with metadata
        eval_summary = {
            "meta": {
                "dataset": args.dataset,
                "model": args.model,
                "num_epochs": args.epochs,
                "num_samples": len(train_dataset),
                "methods": {}
            },
            "binary_scores": eval_dict
        }

        # Add per-method metadata
        for method in args.methods:
            if method == "aum":
                eval_summary["meta"]["methods"][method] = {
                    "reference_source": "https://arxiv.org/pdf/2001.10528",
                    "conversion_method": "https://arxiv.org/pdf/2410.03429"
                }
            elif method == "datamaps":
                eval_summary["meta"]["methods"][method] = {
                    "reference_source": "https://arxiv.org/pdf/2009.10795",
                    "conversion_method": "Dataset divided into 3 equal-sized groups as described in https://arxiv.org/pdf/2009.10795"
                }
            elif method == "el2n":
                eval_summary["meta"]["methods"][method] = {
                    "reference_source": "https://arxiv.org/pdf/2211.05610",
                    "conversion_method": "Samples with scores over a threshold (80% by default) are labelled hard"
                }
            elif method == "forgetting":
                eval_summary["meta"]["methods"][method] = {
                    "reference_source": "",
                    "conversion_method": "Forgotten or never learned samples are labelled hard"
                }
            elif method == "grand":
                eval_summary["meta"]["methods"][method] = {
                    "reference_source": "https://arxiv.org/pdf/2211.05610",
                    "conversion_method": "Samples with scores over a threshold (80% by default) are labelled hard"
                }
            elif method == "loss":
                eval_summary["meta"]["methods"][method] = {
                    "reference_source": "",
                    "conversion_method": "Samples with scores over a threshold (80% by default) are labelled hard"
                }
            elif method == "regularisation":
                eval_summary["meta"]["methods"][method] = {
                    "reference_source": "https://arxiv.org/pdf/2107.09044",
                    "conversion_method": "Misclassified samples are labelled hard"
                }


        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            pickle.dump(eval_summary, temp_file)
            temp_file_path = temp_file.name

        artifact = wandb.Artifact(f"eval_summary_run{run_id}", type="pickle")
        artifact.add_file(temp_file_path, name=f"eval_summary_run{run_id}.pkl")
        wandb.run.log_artifact(artifact)

        # Clean up the temporary file
        os.remove(temp_file_path)

        # add sleep in case of machine latency
        time.sleep(10)

        wandb.finish()

if __name__ == "__main__":
    main()
