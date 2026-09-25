### MAIA 261 - 2D Chest X-Ray classification
## Preprocessing README

# Scope
The preprocessing pipeline resizes and enhances the raw
X-Ray images, builds a Json file linking each preprocessed
image to its label, and splits the dataset into a test
set and stratified cross-validation folds for training. 
All the preprocessed samples are saved under the same path.

# Commands
To run the full pipeline from the project root:
```bash
python preprocessing.py
```

# Steps

- Step 1 : Image resizing and CLAHE contrast enhancement
- Step 2 : Preprocessed dataset Json file creation (paths and labels)
- Step 3 : Stratified train/test split and K-Fold cross-validation splits

# Steps details

- Step 1 :
  Resizing each raw image to a fixed target size while preserving
  its aspect ratio (padding), then applying CLAHE contrast
  enhancement. Files that cannot be opened are skipped and logged.
  The resulting images are saved in data/preprocessed/check-X-ray.
- Step 2 :
  Matching each preprocessed image to its label from the raw CSV,
  checking their integrity again, and saving the paths/labels
  mapping into data/preprocessed/dataset.json.
- Step 3 :
  Splitting the dataset into a fixed, stratified test set and a
  remaining train+val set, based on the ratio set in the configuration 
  file. Applying a stratified K-Fold cross-validation on the remaining
  train+val set, and saving the test set and fold indices into
  data/preprocessed/splits.json.

# Data output architecture

data/preprocessed/
├── check-X-ray/         # resized and CLAHE-enhanced images
├── dataset.json         # {filename: [image_path, label]}
└── splits.json          # {"test": [...], "folds": {fold_num: {"train": [...], "val": [...]}}}

# Important notes

All parameters (target size, CLAHE, split ratio, seed, number of
folds) are set in configs/preprocessing_config.yaml. This
pipeline requires data/raw/ to already contain the raw images
and CSV file.
