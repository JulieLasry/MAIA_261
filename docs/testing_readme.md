### MAIA 261 - Project name

## Testing README

# Scope
The testing pipeline evaluates the trained model on the fixed test
set, across every fold's checkpoint. Standard classification and
clinical metrics are averaged across folds and the ROC/PRC curves,
decision threshold scenarios, MC Dropout uncertainty and latent
space visualization are illustrated using the best-performing fold
only.

# Commands
To run the full evaluation / test pipeline from the project root:
```bash
python testing.py
```

# Steps
- Step 1 : Metrics evaluation across all folds
- Step 2 : Best fold in-depth analysis

# Steps details
- Step 1 :
  Evaluates every fold's checkpoint on the test set, computing the
  classification report (accuracy, precision/recall/F1/F2 per class
  and macro-averaged) and clinical metrics (sensitivity, specificity,
  PPV, NPV per class). Averages accuracy, macro recall, macro F2 and
  bacteria recall across folds.
- Step 2 :
  Identifies the best-performing fold (highest validation AUPRC on
  the bacteria-infected class, from the training results). Then on
  that fold's checkpoint, plots the one-vs-rest ROC and
  Precision-Recall curves (with AUROC/AUPRC) for all classes, the
  confusion matrix, proposes two decision threshold scenarios
  (global performance versus cautious on bacteria false negatives),
  estimates per-sample prediction uncertainty (with the Monte Carlo 
  Dropout technique), and visualizes the 2D latent space (UMAP) which 
  separates correctly and incorrectly predicted samples per class.

# Data output architecture
outputs/test/
├── metrics/
│   ├── test_results.json        # per-fold + averaged test metrics
│   └── best_fold_results.json   # best fold: threshold scenarios, uncertainty
└── figures/
    ├── roc_all_classes.png
    ├── prc_all_classes.png
    ├── confusion_matrix.png
    └── latent_space.png

# Important notes
Requires training.py to have already run (needs the saved
checkpoints and outputs/train/train_results.json). Parameters
(class labels/names, MC Dropout passes, UMAP seed) are set in
configs/testing_config.yaml. Also, the datetime displayed in the
test metrics is in UTC (in France, add +2 hours).
