import lightning as L

class ChestXRayDataModule(L.LightningDataModule):
    """
    Lightning DataModule building the train/val/test Datasets and
    DataLoaders for one specific cross-validation fold.

    Args:
        preprocessing_config (dict): The preprocessing config dict.
        training_config (dict): The training config dict (batch size, num_workers...).
        fold_num (int): Which fold to use for this DataModule instance.
    """

    def __init__(
        self,
        preprocessing_config: dict,
        training_config: dict,
        fold_num: int,
    ) -> None:
        ...

    def setup(self, stage: str | None = None) -> None:
        """
        Loads dataset.json and splits.json, resolves the fold's
        train/val indices and the fixed test indices into actual
        (path, label) samples, and builds the 3 Datasets.
        """
        ...

    def train_dataloader(self):
        """Returns the DataLoader for the training set of this fold."""
        ...

    def val_dataloader(self):
        """Returns the DataLoader for the validation set of this fold."""
        ...

    def test_dataloader(self):
        """Returns the DataLoader for the fixed test set."""
        ...

# Piste pour setup (le point important — reconstruire les samples à partir des indices) :
# dataset_dict = ...  # charger dataset.json
# filenames = list(dataset_dict.keys())  # même ordre que dans DataSplitting

# train_idx = splits["folds"][str(fold_num)]["train"]
# train_samples = [
#     (dataset_dict[filenames[i]][0], dataset_dict[filenames[i]][1])
#     for i in train_idx
# ]
# (rappelle-toi : les clés de folds reviennent en string après un json.load, d'où str(fold_num))
