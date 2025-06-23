import argparse
import numpy as np
import os
import pickle
import random
import yaml
import torch
import wandb
from typing import List

from dataloader import DataLoaderFactory
from trainer import Trainer

AVAILABLE_MODELS: List[str] = [
    "bert-tiny", "bert-base", "bert-large",
    "roberta-base", "roberta-large",
    "xlnet-base", "xlnet-large",
]

AVAILABLE_DATASETS: List[str] = [
    "multi_nli", "civilcomments_wilds", "fever",
    "qqp", "synthetic_mnli_labeled",
]

AVAILABLE_METHODS: List[str] = [
    "aum", "datamaps", "el2n",
    "loss", "forgetting", "grand", "accuracy",
]

# A dictionary containing metadata for each data selection method.
METHOD_METADATA = {
    "aum": {
        "reference_source": "https://arxiv.org/pdf/2410.03429.pdf",
        "conversion_method": "A Gaussian Mixture Model (GMM) is fitted to cluster examples into difficulty levels.",
    },
    "datamaps": {
        "reference_source": "https://arxiv.org/pdf/2009.10795.pdf",
        "conversion_method": "Dataset divided into easy, ambiguous, and hard groups based on confidence and variability.",
    },
    "el2n": {
        "reference_source": "https://arxiv.org/pdf/2211.05610.pdf",
        "conversion_method": "Samples with EL2N scores over a percentile threshold are labelled hard.",
    },
    "forgetting": {
        "reference_source": "https://arxiv.org/pdf/1812.05159.pdf",
        "conversion_method": "Samples forgotten at least once (or never learned) are labelled hard.",
    },
    "grand": {
        "reference_source": "https://arxiv.org/pdf/2211.05610.pdf",
        "conversion_method": "Samples with GraND scores over a percentile threshold are labelled hard.",
    },
    "loss": {
        "reference_source": "https://arxiv.org/pdf/2007.06778",
        "conversion_method": "Samples with the highest loss are labelled hard.",
    },
    "accuracy": {
        "reference_source": None,
        "conversion_method": "Samples with the lowest cumulative accuracy are labelled hard.",
    },
}

def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for the training script."""
    parser = argparse.ArgumentParser(description="Train a transformer model and analyse its training dynamics.")
    parser.add_argument("--model", type=str, choices=AVAILABLE_MODELS, required=True, help="Model architecture to use.")
    parser.add_argument("--dataset", type=str, choices=AVAILABLE_DATASETS, required=True, help="Dataset to use for training and evaluation.")
    parser.add_argument("--methods", nargs="+", choices=AVAILABLE_METHODS, required=True, help="List of data selection methods to analyse.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--num_runs", type=int, default=1, help="Number of times to repeat the experiment with different seeds.")
    parser.add_argument("--wandb_config", type=str, default=None, help="Path to a wandb.yaml file with API key and entity.")
    
    # Hyperparameters
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="Evaluation batch size.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Peak learning rate for the optimiser.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for the optimiser.")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Proportion of training steps for linear warmup.")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", choices=["linear", "cosine", "constant"], help="Learning rate scheduler type.")
    parser.add_argument("--grad_acc_steps", type=int, default=1, help="Number of gradient accumulation steps.")
    parser.add_argument("--percentile", type=int, default=80, help="Percentile threshold for identifying 'hard' examples.")
    parser.add_argument("--beta1", type=float, default=0.9, help="AdamW beta1 parameter.")
    parser.add_argument("--beta2", type=float, default=0.999, help="AdamW beta2 parameter.")
    parser.add_argument("--eps", type=float, default=1e-8, help="AdamW epsilon parameter.")
    
    return parser.parse_args()
    
def main():
    """
    Main function to orchestrate the training and evaluation process.
    """
    args = parse_args()

    if args.wandb_config:
        with open(args.wandb_config, "r") as f:
            wandb_creds = yaml.safe_load(f)
        os.environ["WANDB_API_KEY"] = wandb_creds.get("wandb_key")
        os.environ["WANDB_ENTITY"] = wandb_creds.get("entity")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    for run_id in range(args.num_runs):
        # Generate a new random seed for each run for independent experiments.
        seed = random.randint(0, 1_000_000)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        print(f"\n--- Starting Run {run_id + 1}/{args.num_runs} (Seed: {seed}) ---")

        # Initialise data loader and fetch all data components.
        data_loader = DataLoaderFactory(args.dataset, args.model)
        dataset_bundle = data_loader.prepare_datasets()

        # Start a new W&B run for each experiment.
        wandb_run = wandb.init(
            project=f"{args.model.replace('/', '-')}-{args.dataset}-analysis",
            name=f"run-{run_id+1}-seed-{seed}",
            config=vars(args)
        )
        
        # The Trainer handles the main training loop.
        trainer = Trainer(
            args=args,
            dataset_bundle=dataset_bundle,
            methods=args.methods,
            eval_metrics=["accuracy", "f1", "precision", "recall"]
        )
        trainer.train()

        # After training, get all the raw statistics.
        stats = trainer.get_unified_stats()
        
        # Prepare a summary dictionary for logging as a W&B artifact.
        summary_artifact = {
            "meta": {
                "dataset": args.dataset,
                "model": args.model,
                "num_epochs": args.epochs,
                "num_samples": len(dataset_bundle.train_dataset),
                "seed": seed,
                "methods": {m: METHOD_METADATA.get(m, {}) for m in args.methods}
            },
            "raw_scores": stats,
        }
        
        # Save the summary as a pickle file and log it as a W&B artifact.
        artifact_name = f"summary_run_{run_id+1}"
        with open(f"{artifact_name}.pkl", "wb") as f:
            pickle.dump(summary_artifact, f)
        
        artifact = wandb.Artifact(artifact_name, type="analysis_results")
        artifact.add_file(f"{artifact_name}.pkl")
        wandb_run.log_artifact(artifact)
        
        wandb.finish()
        
        # Clean up the local pickle file after logging.
        os.remove(f"{artifact_name}.pkl")

if __name__ == "__main__":
    main()
