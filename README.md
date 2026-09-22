### MAIA 261 - Project name

## Project scope : 
"TO COMPLETE"

## Installation

Requires Python 3.10.12

**macOS / Linux** "TO COMPLETE"
```bash
git clone <"GithubRepo">
cd Bone_Age_Classification_Regression
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows** "TO COMPLETE"
```powershell
git clone <https://github.com/JulieLasry/rsna-boneage-classification-regression>
cd Bone_Age_Classification_Regression
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**Data.** Data coming from the "TO COMPLETE":

```
data/raw/ "TO COMPLETE"
├── boneage-training-dataset.csv
└── boneage-training-dataset/boneage-training-dataset/*.png
```

## Execution and usage

Run the data pipeline from the project root:

- For data processing : "TO COMPLETE"
```bash
python processing.py
```
This creates `data/processed/data.csv` (fold and test assignment for each / "TO COMPLETE"
image) and prints the size of the train / val / test sets for every fold.


Parameters are set in `configs/`: "TO COMPLETE"

| File | Content |
|------|---------| "TO COMPLETE"
| `preprocessing_config.yaml` | task (`classification` / `regression`), paths, image size, split, augmentations, normalization |
| `training_config.yaml` | dataloader and training parameters |

Project structure:
 "TO COMPLETE"
```
├── configs/           # YAML configs
├── data/              # raw and processed data (not versioned)
├── models/            # trained weights (not versioned)
├── notebooks/         # data exploration
├── src/data/          # raw loading, transforms, Dataset, DataModule
├── processing.py      # runs the data pipeline
└── train.py, test.py, inference.py, optimize.py
```

## Used technologies "TO COMPLETE"

| Technology | Explanation |
|---|---|
| PyTorch | Deep learning framework |
| Lightning | Structures the data and training code (`DataModule`, `LightningModule`) |
| Albumentations | Fast image augmentations and preprocessing |
| scikit-learn | Stratified splits and K-fold cross-validation |
| pandas / NumPy | Tabular data handling |
| Pillow | Image loading |
| PyYAML | Config files |

Planned: MLflow (experiment tracking), Optuna (hyperparameter search).

## Current features "TO COMPLETE"

- Config-driven pipeline, with no hard-coded hyperparameters
- Classification or regression target, chosen in the config
- Stratified test set, then stratified 5-fold cross-validation
- Train-only augmentations (flip, rotation, affine, noise, brightness/contrast)
- One Lightning `DataModule` per fold

## Contributing "TO COMPLETE"

This is a personal learning project, but suggestions are welcome:

1. Fork the repository and create a branch (`git checkout -b feature/my-idea`).
2. Commit your changes with a clear message.
3. Open a pull request describing the change.

## Contributors

<Julie Lasry> (author)

## Author

<Julie Lasry> — <juliexlasry@gmail.com>

## Change log

- **v0.1.0** — Initial data pipeline: raw loading, split, transforms,
  Dataset and DataModule.

