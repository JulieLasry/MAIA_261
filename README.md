### MAIA 261 - 2D Chest X-Ray classification

## Project scope : 
The aim of this technical interview project
is to classify thoracic 2D X-Ray from children
aged 1 to 5 years into the three following categories:
- Normal
- Bacteria-infected
- Virus-infected
This challenge will then highlight my ability to
analyze and preprocess the data, wisely choose a 
Deep Learning classifier model, train and test it, 
and select appropriate metrics and prediction 
representations. Optional work will focus on unsupervised 
methods for new labels detection, and on insightful tools 
to provide a better clinical understanding of the model 
predictions.

## Installation

Requires Python 3.10.12

**macOS / Linux**
```bash
git clone https://github.com/JulieLasry/MAIA_261.git
cd MAIA_261
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows**
```powershell
git clone https://github.com/JulieLasry/MAIA_261.git
cd MAIA_261
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Data
Please place the raw data provided as part of this technical interview,
as unzipped files, under data/raw/ as follows:

data/raw/
├── data_info.csv
└── check-X-ray/
    └── *.tiff

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

