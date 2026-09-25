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

Run the preprocessing pipeline from the project root:

```bash
python preprocessing.py
```
This resizes and enhances (CLAHE) all raw images, saves them to
data/preprocessed/check-X-ray/, builds dataset.json (image
paths and labels), splits the data into a stratified test set and
K-Fold cross-validation folds, and saves the split
indices to splits.json.

Training (python training.py) is in progress and will train the
classifier for each fold, log metrics and artifacts to MLflow.

Parameters are set in configs/:

┌───────────────────────────┬───────────────────────────────────────────────────────────────────────────────────────────┐
│           File            │                                          Content                                          │
├───────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────┤
│ preprocessing_config.yaml │ raw/output paths, target image size, CLAHE parameters, split ratio, seed, number of folds │
├───────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────┤
│ training_config.yaml      │ augmentations, normalization, dataloader parameters                                       │
└───────────────────────────┴───────────────────────────────────────────────────────────────────────────────────────────┘

Project structure:
├── configs/               # YAML configs (preprocessing, training)
├── data/                  # raw and preprocessed data (not versioned)
├── docs/                  # sub-readmes (notebooks, preprocessing)
├── models/                # trained weights (not versioned)
├── notebooks/             # data exploration
├── src/data/               # preprocessing, splitting, augmentation, Dataset, DataModule
├── src/models/             # model architecture
├── src/utils/              # shared utilities
├── preprocessing.py        # runs the preprocessing + splitting pipeline
└── training.py              # runs the training pipeline (in progress)

Used technologies

┌───────────────────┬──────────────────────────────────────────────────────────────────────┐
│    Technology     │                             Explanation                              │
├───────────────────┼──────────────────────────────────────────────────────────────────────┤
│ PyTorch           │ Deep learning framework                                              │
├───────────────────┼──────────────────────────────────────────────────────────────────────┤
│ Lightning         │ Structures the data and training code (DataModule, LightningModule)  │
├───────────────────┼──────────────────────────────────────────────────────────────────────┤
│ MONAI             │ Medical-imaging-specific model architectures (EfficientNet backbone) │
├───────────────────┼──────────────────────────────────────────────────────────────────────┤
│ Albumentations    │ Fast image augmentations and preprocessing                           │
├───────────────────┼──────────────────────────────────────────────────────────────────────┤
│ OpenCV            │ CLAHE contrast enhancement                                           │
├───────────────────┼──────────────────────────────────────────────────────────────────────┤
│ scikit-learn      │ Stratified splits and K-fold cross-validation                        │
├───────────────────┼──────────────────────────────────────────────────────────────────────┤
│ pandas / NumPy    │ Tabular data handling                                                │
├───────────────────┼──────────────────────────────────────────────────────────────────────┤
│ Pillow / tifffile │ Image loading                                                        │
├───────────────────┼──────────────────────────────────────────────────────────────────────┤
│ PyYAML            │ Config files                                                         │
└───────────────────┴──────────────────────────────────────────────────────────────────────┘

Planned: MLflow (experiment tracking). Optuna (hyperparameter
search) is included as a dependency but not used yet.

Current features

- Config-driven preprocessing pipeline, no hard-coded hyperparameters
- CLAHE contrast enhancement and aspect-ratio-preserving resizing
- Corrupted/invalid file detection before and after preprocessing
- Stratified test set, then stratified K-Fold cross-validation
- Train-only augmentations (flip, affine, noise, brightness/contrast, blur)
- One Lightning DataModule per fold

## Contributors

<Julie Lasry> (author)

## Author

<Julie Lasry> — <juliexlasry@gmail.com>

## Change log

- **v0.1.0** — Initial data pipeline: raw loading, split, transforms,
  Dataset and DataModule.

