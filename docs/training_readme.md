### MAIA 261 - Project name

## Training README

# Scope
The training pipeline trains a ResNet-50 classifier (ImageNet-
pretrained) across each stratified cross-validation fold, in two
phases (frozen feature extractor, then progressive fine-tuning),
logging metrics and artifacts to MLflow.

# Commands
To run the full training pipeline from the project root:
```bash
python training.py
```

# Steps
- Step 1 : Dataloader length validation per fold
- Step 2 : Training of each fold, in two phases

# Steps details
- Step 1 :
  Builds the DataModule for each fold and prints the length of its
  train/val/test datasets, to validate the data/split before training.
- Step 2 :
  For each fold, trains the model in two phases. Phase 1 has its 
  backbone frozen (only the classification head trainable), and
  phase 2 has the last backbone stages progressively unfrozen for
  fine-tuning, at a lower learning rate. Tracks the best validation
  AUPRC (bacteria-infected class) across both phases, saves the
  corresponding checkpoint, and logs metrics/artifacts to MLflow.
  Averages the key validation metrics (macro recall, macro F2,
  bacteria recall, bacteria f2) and loss across all folds.

# Data output architecture

outputs/train/
├── mlflow/                    # MLflow tracking databases, per fold
├── train_results.json         # per-fold + averaged validation metrics
└── execution_results.json     # per-fold training time and parameter count

models/checkpoint/per_fold/
└── {run_name}_fold_{n}.ckpt   # best checkpoint per fold (+ last.ckpt)

Important notes

All hyperparameters (learning rates, epochs, dropout, optimizer,
early stopping patience, unfreezing number of layers) are set in
configs/training_config.yaml. Change training.name before a
new run to keep previous checkpoints/logs instead of overwriting
them. Requires data/preprocessed/ to already exist for training 
(run preprocessing.py first). Also, the datetime displayed in the
train metrics is in UTC (in France, add +2 hours).
