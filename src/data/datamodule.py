import lightning as L
from typing import Optional
from pathlib import Path
import json
from torch.utils.data import DataLoader

from src.data.dataset import ChestXRayDataset

class ChestXRayDataModule(L.LightningDataModule):
    """
    Lightning DataModule building the train/val/test Datasets
    and DataLoaders for one specific cross-validation fold
    in the training set.

    Args:
        preprocessing_config (dict): The dict of the preprocessing 
        main paths and variable values in preprocess.
        training_config (dict): The dict of the training main paths 
        and variable values in training process.
        fold_num (int): The fold number to use in the Dataloader.
    """

    def __init__(
        self,
        preprocessing_config: dict,
        training_config: dict,
        fold_num: int,
    ) -> None:
        super().__init__()
        self.preprocessing_config = preprocessing_config
        self.training_config = training_config
        self.fold_num = fold_num

        self.train_dataset: Optional[ChestXRayDataset]  = None
        self.val_dataset: Optional[ChestXRayDataset]  = None
        self.test_dataset: Optional[ChestXRayDataset]  = None
        

    def setup(self, stage: str | None = None) -> None:
        """
        Loads dataset.json and splits.json, resolves the fold's
        train/val indices and the fixed test indices into actual
        (path, label) samples, and builds the 3 Datasets.
        """
        output_dir: Path = Path(self.preprocessing_config['paths']['output_dir'])
        dataset_json: Path = output_dir / "dataset.json"
        splits_json: Path = output_dir / "splits.json"

        # Loading the Json files as dictionnaries
        with open(dataset_json, 'r', encoding='utf-8') as dataset_file:
            dataset_dict: dict = json.load(dataset_file)
        with open(splits_json, 'r', encoding='utf-8') as split_file:
            splits_dict: dict = json.load(split_file)

        # Retrieving the train/val/test indices (with the train/val fold)
        filenames: list[str] = [i for i in dataset_dict.keys()]
        fold_key: str = str(self.fold_num)
        train_idx: list[int] = splits_dict["folds"][fold_key]["train"]
        val_idx: list[int] = splits_dict["folds"][fold_key]["val"]
        test_idx: list[int] = splits_dict["test"]

        # Constructing the train/val/test samples
        self.train_dataset = ChestXRayDataset(
            samples=[(dataset_dict[filenames[i]][0], dataset_dict[filenames[i]][1]) 
                    for i in train_idx],
            training_config=self.training_config,
            is_train=True
        )

        self.val_dataset = ChestXRayDataset(
            samples=[(dataset_dict[filenames[i]][0], dataset_dict[filenames[i]][1]) 
                    for i in val_idx],
            training_config= self.training_config,
            is_train=False
        )

        self.test_dataset = ChestXRayDataset(
            samples=[(dataset_dict[filenames[i]][0], dataset_dict[filenames[i]][1]) 
                    for i in test_idx],
            training_config= self.training_config,
            is_train=False
        )

    def train_dataloader(self):
        """
        Returns the DataLoader for the training set of this fold.
        """
        assert self.train_dataset is not None, \
            "train_dataset is None. Call setup()"
        return DataLoader(
            self.train_dataset,
            batch_size=self.training_config['dataloader']['batch_size'],
            num_workers=self.training_config['dataloader']['num_workers'],
            shuffle=True
        )
        

    def val_dataloader(self):
        """
        Returns the DataLoader for the validation set of this fold.
        """
        assert self.val_dataset is not None, \
            "val_dataset is None. Call setup()"
        return DataLoader(
             self.val_dataset,
             batch_size=self.training_config['dataloader']['batch_size'],
             num_workers=self.training_config['dataloader']['num_workers'],
             shuffle=False
         )
        

    def test_dataloader(self):
        """
        Returns the DataLoader for the fixed test set.
        """
        assert self.test_dataset is not None, \
            "test_dataset is None. Call setup()"
        return DataLoader(
             self.test_dataset,
             batch_size=self.training_config['dataloader']['batch_size'],
             num_workers=self.training_config['dataloader']['num_workers'],
             shuffle=False
         )
