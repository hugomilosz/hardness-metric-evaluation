# Hardness Metric Evaluation

> A PyTorch framework for evaluating data-centric AI metrics to identify hard and noisy examples in NLP datasets.

This repository provides an experimental pipeline built on top of Hugging Face `transformers` and `accelerate` to track model training dynamics. By calculating various "hardness" metrics during the training loop, this tool helps identify mislabeled, ambiguous, or difficult examples within a dataset.

## Features

- **8 Data-Centric Tracking Methods:** Implements AUM, DataMaps, EL2N, Forgetting, GraNd, Loss, Accuracy, and Regularisation.
- **Distributed Training Ready:** Wrapped in Hugging Face `accelerate` for seamless mixed-precision (FP16) and multi-GPU training.
- **Experiment Tracking:** Integration with Weights & Biases (WandB) to log training metrics and export data-centric artifacts automatically.
- **NLP Models:** Run experiments on BERT, RoBERTa, and XLNet architectures.

## Supported Architectures & Datasets

| **Models** | **Datasets** |
| :--- | :--- |
| `bert-tiny`, `bert-base`, `bert-large` | `multi_nli`, `synthetic_mnli_labeled` |
| `roberta-base`, `roberta-large` | `civilcomments_wilds` |
| `xlnet-base`, `xlnet-large` | `fever`, `qqp`, `toy` |

## Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/hugomilosz/hardness-metric-evaluation.git](https://github.com/hugomilosz/hardness-metric-evaluation.git)
   cd hardness-metric-evaluation
   ```

2. Install the required dependencies:
   ```bash
   pip install torch numpy transformers accelerate evaluate wandb pyyaml
   ```

3. Configure Weights & Biases (Optional but recommended):
   Create a `wandb.yaml` file in the root directory with your credentials, or log in via the CLI:
   ```bash
   wandb login
   ```

## Usage

You can launch a training run using the `main.py` script. You must specify the model, dataset, and the evaluation methods you want to track.

### Example Command

```bash
python main.py \
  --model roberta-base \
  --dataset fever \
  --methods aum datamaps el2n grand \
  --epochs 3 \
  --batch_size 32 \
  --learning_rate 2e-5 \
  --fp16
```

### Key Arguments
* `--model`: The Hugging Face model architecture to use.
* `--dataset`: The dataset to train and evaluate on.
* `--methods`: A space-separated list of hardness metrics to track (e.g., `aum datamaps el2n forgetting grand loss accuracy regularisation`).
* `--epochs`: Number of training epochs (default: 3).
* `--fp16`: Flag to enable mixed-precision training for faster execution.
* `--wandb_config`: Path to your `wandb.yaml` file for logging.

*(Run `python main.py --help` for a full list of hyperparameters and optimiser settings).*

## Evaluation Methods

This framework tracks training dynamics using the following methodologies:

* **AUM (Area Under the Margin):** Tracks the difference between the assigned class logit and the highest alternative class logit.
* **DataMaps:** Groups examples into easy, ambiguous, and hard regions based on confidence and variability.
* **EL2N (Error L2-Norm):** Measures the L2 distance between the model's predicted probabilities and the one-hot encoded labels.
* **Forgetting Events:** Tracks how often a previously correctly classified example is misclassified in subsequent epochs.
* **GraNd (Gradient Normed):** Uses the expected gradient norm to identify highly informative training examples.
* **Loss/Accuracy:** Traditional tracking of highest loss or lowest accuracy examples across training.

## Output

Once the training loop concludes, the `Trainer` consolidates the tracked statistics. If WandB is enabled, the unified stats are packed into a `.pkl` file and logged as a `dataset_scores` artifact directly to your WandB dashboard for further analysis.
