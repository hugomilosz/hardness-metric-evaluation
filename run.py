import argparse
import numpy as np
import os
import pickle
import random
import yaml
import torch
import wandb

from dataloader import DataLoader
from trainer import Trainer
from transformers import TrainingArguments

AVAILABLE_MODELS = [
    "bert-tiny", "bert-base", "bert-large",
    "roberta-base", "roberta-large",
    "xlnet-base", "xlnet-large",
]

AVAILABLE_DATASETS = [
    "multi_nli", "civilcomments_wilds", "fever",
    "qqp", "toy", "synthetic_mnli_labeled",
]

AVAILABLE_METHODS = [
    "aum", "datamaps", "el2n", "loss", "forgetting",
    "grand", "accuracy", "regularisation"
]

METHOD_METADATA = {
    "aum": {
        "reference_source": "https://arxiv.org/pdf/2410.03429.pdf",
        "conversion_method": "A Gaussian Mixture Model (GMM) is fitted to training dynamics (confidence, variability, correctness) to cluster examples into difficulty levels.",
    },
    "datamaps": {
        "reference_source": "https://arxiv.org/pdf/2009.10795.pdf",
        "conversion_method": "Dataset divided into easy, ambiguous, and hard groups based on model learning dynamics (confidence, variability, correctness).",
    },
    "el2n": {
        "reference_source": "https://arxiv.org/pdf/2211.05610.pdf",
        "conversion_method": "Samples with the highest L2 norm between logits and one-hot labels (top N%) are labelled as hard.",
    },
    "forgetting": {
        "reference_source": "https://arxiv.org/pdf/1812.05159.pdf",
        "conversion_method": "Samples that the model misclassifies after having correctly classified them in an earlier epoch are labelled hard.",
    },
    "grand": {
        "reference_source": "https://arxiv.org/pdf/2211.05610.pdf",
        "conversion_method": "Samples with the highest gradient norm scores (GraNd) (top N%) are labelled as hard.",
    },
    "loss": {
        "reference_source": "https://arxiv.org/pdf/2007.06778",
        "conversion_method": "Samples with the highest training loss (top N%) are labelled as hard.",
    },
    "accuracy": {
        "reference_source": "https://arxiv.org/pdf/2007.06778",
        "conversion_method": "Samples with the lowest prediction accuracy across epochs (top N%) are labelled as hard.",
    },
    "regularisation": {
        "reference_source": "https://arxiv.org/pdf/2107.09044.pdf",
        "conversion_method": "Samples that are misclassified during training are labelled as hard.",
    }
}

def parse_args():
    """Parses command-line arguments for the training run."""
    parser = argparse.ArgumentParser(description="Train a transformer model and analyze data characteristics.")
    
    # Run Config
    parser.add_argument("--model", type=str, choices=AVAILABLE_MODELS, required=True, help="Model name from Hugging Face.")
    parser.add_argument("--dataset", type=str, choices=AVAILABLE_DATASETS, required=True, help="Dataset name to use.")
    parser.add_argument("--methods", nargs="+", choices=AVAILABLE_METHODS, required=True, help="List of data analysis methods to apply.")
    parser.add_argument("--num_runs", type=int, default=1, help="Number of repeated runs for training and evaluation.")
    
    # Training Hyperparameters
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Per-device training batch size.")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="Per-device evaluation batch size.")
    parser.add_argument("--grad_acc_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--percentile", type=int, default=80, help="Percentile threshold for identifying 'hard' examples in applicable methods.")
    parser.add_argument("--fp16", action="store_true", help="Enable mixed-precision training (FP16).")
    parser.add_argument("--dataloader_num_workers", type=int, default=4, help="Number of workers for the dataloader.")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd", "adafactor"], help="The optimizer to use for training.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Initial learning rate for the optimizer.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for the optimizer.")
    parser.add_argument("--beta1", type=float, default=0.9, help="AdamW optimizer's beta1 parameter.")
    parser.add_argument("--beta2", type=float, default=0.999, help="AdamW optimizer's beta2 parameter.")
    parser.add_argument("--eps", type=float, default=1e-8, help="AdamW optimizer's epsilon parameter.")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for the SGD optimizer.")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", choices=[
        "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"
    ], help="Learning rate scheduler type.")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Ratio of total training steps for the learning rate warmup.")
    parser.add_argument("--wandb_config", type=str, default=None, help="Path to wandb.yaml with API key and entity.")

    return parser.parse_args()
    
def main():
    """Main function to orchestrate the training and evaluation process."""
    args = parse_args()

    if args.wandb_config:
        with open(args.wandb_config, "r") as f:
            wandb_creds = yaml.safe_load(f)
        os.environ["WANDB_API_KEY"] = wandb_creds["wandb_key"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    for run_id in range(args.num_runs):
        print(f"\n--- Starting Run {run_id + 1}/{args.num_runs} ---")
        torch.cuda.empty_cache()
        seed = random.randint(0, 1_000_000)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        data_loader = DataLoader(args.dataset, args.model)
        dataset_bundle = data_loader.prepare_datasets()

        training_config = TrainingArguments(
            output_dir=f"./results/{args.model.replace('/', '-')}-{args.dataset}-run{run_id}",
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            gradient_accumulation_steps=args.grad_acc_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            adam_beta1=args.beta1,
            adam_beta2=args.beta2,
            adam_epsilon=args.eps,
            lr_scheduler_type=args.lr_scheduler_type,
            warmup_ratio=args.warmup_ratio,
            seed=seed,
            fp16=args.fp16,
            dataloader_num_workers=args.dataloader_num_workers,
            save_strategy="no",
            evaluation_strategy="epoch",
            report_to="none",
        )
        
        # Add custom args to the config
        training_config.optimizer = args.optimizer
        training_config.momentum = args.momentum

        wandb.init(
            project=f"{args.model.replace('/', '-')}_{args.dataset}_analysis",
            name=f"{args.optimizer}_run_{run_id+1}",
            config=training_config.to_dict(),
        )
        wandb.config.update({
            "model_name": args.model,
            "dataset_name": args.dataset,
            "analysis_methods": args.methods,
            "device": str(device),
        })

        trainer = Trainer(
            args=training_config,
            dataset_bundle=dataset_bundle,
            methods=args.methods,
            eval_metrics=["accuracy", "f1", "precision", "recall"]
        )
        trainer.train()
        
        stats = trainer.get_unified_stats()
        
        eval_summary = {
            "meta": {
                "dataset": args.dataset,
                "model": args.model,
                "num_epochs": args.epochs,
                "num_samples": len(dataset_bundle.train_dataset),
                "optimizer": args.optimizer,
                "methods": {method: METHOD_METADATA.get(method, {}) for method in args.methods}
            },
            "raw_scores": stats
        }

        # WandB Logging and Cleanup
        target_dir = ""
        os.makedirs(target_dir, exist_ok=True)
        filename = f"summary_{wandb.run.id}.pkl"
        temp_file_path = os.path.join(target_dir, filename)

        with open(temp_file_path, "wb") as f:
            pickle.dump(eval_summary, f)

        artifact = wandb.Artifact(f"summary_run_{run_id+1}", type="dataset_scores")
        artifact.add_file(temp_file_path, name=filename)
        wandb.run.log_artifact(artifact)
        artifact.wait()

        os.remove(temp_file_path)
        wandb.finish()

        del trainer
        del dataset_bundle.model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()