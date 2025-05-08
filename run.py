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
from trainer import Trainer  # Import the new ImprovedTrainer
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
    "roberta-large",
    "xlnet-base",
    "xlnet-large",
]

AVAILABLE_DATASETS = [
    "multi_nli",
    "civilcomments_wilds",
    "fever",
    "qqp",
    "toy",
]

AVAILABLE_METHODS = [
    "aum",
    "datamaps",
    "el2n",
    "loss",
    "forgetting",
    "grand",
    "accuracy"
]

MODEL_CONFIGS = {
    "bert-tiny": {
        "learning_rate": 5e-5,
        "weight_decay": 0.01,
        "warmup_ratio": 0.06,
        "gradient_accumulation_steps": 1
    },
    "bert-base": {
        "learning_rate": 3e-5,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "gradient_accumulation_steps": 2
    },
    "bert-large": {
        "learning_rate": 2e-5,
        "weight_decay": 0.02,
        "warmup_ratio": 0.1,
        "gradient_accumulation_steps": 4
    },
    "roberta-base": {
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "warmup_ratio": 0.06,
        "gradient_accumulation_steps": 2
    },
    "roberta-large": {
        "learning_rate": 1e-5,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "gradient_accumulation_steps": 4
    },
    "xlnet-base": {
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "gradient_accumulation_steps": 2
    },
    "xlnet-large": {
        "learning_rate": 1e-5,
        "weight_decay": 0.02,
        "warmup_ratio": 0.1,
        "gradient_accumulation_steps": 4
    }
}

METHOD_METADATA = {
    "aum": {
        "reference_source": "https://arxiv.org/pdf/2410.03429.pdf",
        "conversion_method": (
            "Training dynamics (confidence, variability, correctness, AUM) are extracted "
            "across normal (Premise+Hypothesis) and hypothesis-only training runs. "
            "A Gaussian Mixture Model (GMM) is fitted to cluster examples into difficulty levels, "
            "avoiding manual thresholds. Based on extended Data Maps methodology."
        ),
    },
    "datamaps": {
        "reference_source": "https://arxiv.org/pdf/2009.10795.pdf",
        "conversion_method": (
            "Dataset divided into 3 equal-sized groups (easy, ambiguous, hard) "
            "based on model learning dynamics (confidence, variability, correctness)."
        ),
    },
    "el2n": {
        "reference_source": "https://arxiv.org/pdf/2211.05610.pdf",
        "conversion_method": (
            "Samples with EL2N (L2 norm between logits and labels) scores over "
            "a threshold (default: top 20%) are labelled hard."
        ),
    },
    "forgetting": {
        "reference_source": "",
        "conversion_method": "Samples forgotten at least once (or never learned) are labelled hard."
    },
    "grand": {
        "reference_source": "https://arxiv.org/pdf/2211.05610.pdf",
        "conversion_method": (
            "Samples with GRAND (gradient norm) scores over "
            "a threshold (default: top 20%) are labelled hard."
        ),
    },
    "loss": {
        "reference_source": "https://arxiv.org/pdf/2007.06778",
        "conversion_method": "Samples with highest loss (default: top 20%) are labelled hard."
    },
    "accuracy": {
        "reference_source": "https://arxiv.org/pdf/2007.06778",
        "conversion_method": "Samples with lowest accuracy (default: top 20%) are labelled hard."
    },
    "regularisation": {
        "reference_source": "https://arxiv.org/pdf/2107.09044.pdf",
        "conversion_method": "Misclassified samples during training are labelled hard."
    }
}

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
    parser.add_argument("--percentile", type=int, default=80, help="Percentile threshold for hard examples (used in applicable methods)")
    return parser.parse_args()

def compute_metrics(eval_pred):
    metric = load("accuracy")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)
    
def main():
    torch.cuda.empty_cache()
    args = parse_args()

    if args.wandb_config:
        with open(args.wandb_config, "r") as f:
            wandb_creds = yaml.load(f, Loader=yaml.FullLoader)
        os.environ["WANDB_API_KEY"] = wandb_creds["wandb_key"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    for run_id in range(args.num_runs):
        seed = 33 + run_id  # change seed for each run
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Load dataset_bundle
        data_loader = DataLoader(args.dataset, args.model)
        dataset_bundle = data_loader.prepare_datasets()
        train_dataset = dataset_bundle.train_dataset
        eval_dataset = dataset_bundle.eval_dataset
        num_labels = dataset_bundle.num_labels

        # Use model configuration from MODEL_CONFIGS if available
        model_config = MODEL_CONFIGS.get(args.model, {
            "learning_rate": 2e-5,
            "weight_decay": 0.01,
            "warmup_ratio": 0.1,
            "gradient_accumulation_steps": 1
        })

        # Training arguments with model-specific settings
        training_args = TrainingArguments(
            output_dir=f"{args.model.replace('/', '-')}-{args.dataset}-run{run_id}",
            learning_rate=model_config["learning_rate"],
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            num_train_epochs=args.epochs,
            weight_decay=model_config["weight_decay"],
            evaluation_strategy="epoch",
            save_strategy="epoch",
            fp16=torch.cuda.is_available(),
            seed=seed,
            report_to="none",
            lr_scheduler_type="linear",
            warmup_ratio=model_config["warmup_ratio"],
            gradient_accumulation_steps=model_config["gradient_accumulation_steps"],
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
                **model_config,  # Include model-specific configs
            },
        )

        # Create model
        model = dataset_bundle.model

        # Use the new ImprovedTrainer
        trainer = Trainer(
            args=training_args,
            dataset_bundle=dataset_bundle,
            methods=args.methods,
            device=device,
            eval_metrics = ["accuracy", "f1", "precision", "recall"]
        )
        trainer.train()

        # Evaluate on test set if available
        if hasattr(dataset_bundle, 'test_dataset') and dataset_bundle.test_dataset is not None:
            test_results = trainer.evaluate(dataset_bundle.test_dataset)
            wandb.log({"test": test_results})
            print(f"Test results: {test_results}")

        # Get stats and evaluate
        stats = trainer.get_unified_stats()
        # evaluator = Evaluator(len(train_dataset), stats, percentile=args.percentile)
        # eval_dict = evaluator.binary_scores

        # Store binary scores with metadata
        eval_summary = {
            "meta": {
                "dataset": args.dataset,
                "model": args.model,
                "num_epochs": args.epochs,
                "num_samples": len(train_dataset),
                "methods": {}
            },
            "raw_scores": stats
        }

        # "binary_scores": eval_dict,

        for method in args.methods:
            eval_summary["meta"]["methods"][method] = METHOD_METADATA.get(method, {})

        with tempfile.NamedTemporaryFile(delete=False, mode="wb") as temp_file:
            pickle.dump(eval_summary, temp_file)
            temp_file_path = temp_file.name
            temp_file.flush()  # Make sure all the data is written to disk
            temp_file.close() 

        # Explicitly close or flush (this is now done by exiting the `with` block)
        artifact = wandb.Artifact(f"eval_summary_run{run_id}", type="pickle")
        artifact.add_file(temp_file_path, name=f"eval_summary_run{run_id}.pkl")
        wandb.run.log_artifact(artifact)

        artifact.wait()  # <-- ensures artifact is fully uploaded


        with open(temp_file_path, "rb") as f:
            try:
                loaded_data = pickle.load(f)
                print("Pickle file loaded successfully, file seems valid")
            except Exception as e:
                print(f"Error loading pickle file: {e}")

        # Clean up the temporary file
        os.remove(temp_file_path)
        
        wandb.finish()

if __name__ == "__main__":
    main()