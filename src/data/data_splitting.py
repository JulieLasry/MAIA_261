from pathlib import Path
import json
from sklearn.model_selection import train_test_split, StratifiedKFold

from src.utils.common_utils import CommonUtils

class DataSplitting:
    """ 
    Loading the image paths and labels coming from the 
    Json dataset file. The files are then split into 
    train and test categories. A Stratified K-Fold Cross
    Validation is then applied on the training set. Finally,
    the splits (train/val/test) are saved in a new Json file. 
    """

    @staticmethod
    def __load_dataset(
        dataset_json_path: Path
    ) -> dict[str, list]:
        """
        Load the Json dataset file, containing image labels and 
        paths. It reads its content and returns the dictionnary
        of all the images paths and labels, further analyzed.

        Args:
            dataset_json_path (Path): The path of the dataset Json file
            with image labels and paths. 

        Returns:
            dict[str, list]
            The dictionnary all images paths and labels.
        """
        # Convert the Json file into a dictionnary
        with open(dataset_json_path, "r",  encoding="utf-8") as dataset_json:
            images_labels_paths: dict[str, list] = json.load(dataset_json)
        return images_labels_paths


    @staticmethod
    def split_train_test(
        dataset_json_path: Path,
        preprocessing_config: dict
    ) -> tuple[list[int]]:
        """
        Split the dataset into stratitifed training and 
        testing sets. It then returns the indices of the 
        remaining train+val and test sets, and the corresponding
        labels.

        Args:
            dataset_json_path (Path): The path of the dataset Json file
            with image labels and paths. 
            preprocessing_config (Path): The dict of the preprocessing main paths 
            and variable values in preprocess. 

        Returns:
            tuple[list[int]]
            The tuple of the remaining train+val and test indices and the associated labels.
        """
        dataset_dict: dict[str, list] = DataSplitting.__load_dataset(
            dataset_json_path=dataset_json_path
        )

        # Obtaining the filenames and labels in the dataset dict
        filenames: list[str] = list(dataset_dict.keys())
        labels: list[int] = [dataset_dict[filename][1] for filename in filenames]
        indices: list[int] = list(range(len(filenames)))

        train_val_indices, test_indices =  train_test_split(
            indices,
            test_size=preprocessing_config['split']['test_size'],
            random_state=preprocessing_config['split']['seed'],
            stratify=labels
        )
        return train_val_indices, test_indices, labels

    @staticmethod
    def create_cv_folds(
        preprocessing_config: dict,
        train_val_indices: list[int], 
        labels: list[int]
    ) -> dict[int, dict[str, list[int]]]:
        """
        Create stratified k-fold cross-validation splits on the
        remaining train+val indices. For each fold, it computes 
        the corresponding train and validation indices.

        Args:
            preprocessing_config (dict): The dict of the preprocessing main paths
            and variable values in preprocess.
            train_val_indices (list): The indices of the train+val set,
            excluding the test set.
            labels (list): The values of all the full dataset.

        Returns:
            dict[int, dict[str, list[int]]]
            The dictionary of train/val indices for each fold.
        """
        cv_splitter = StratifiedKFold(
            n_splits=preprocessing_config['split']['n_folds'], 
            random_state=preprocessing_config['split']['seed'], 
            shuffle=True
        )
        train_val_labels: list[int] = [labels[i] for i in train_val_indices]
        folds: dict[int, dict[str, list[int]]] = {}

        for fold_num, (train_idx, val_idx) in enumerate(
            cv_splitter.split(train_val_indices, train_val_labels), 1
        ):
            folds[fold_num] = {
                "train": [train_val_indices[i] for i in train_idx],
                "val": [train_val_indices[i] for i in val_idx]
            }
        return folds


    @staticmethod
    def save_splits(
        preprocessing_config: dict,
        test_indices: list[int],
        folds: dict[int, dict[str, list[int]]]
    ) -> None:
        """
        Save the test set indices and the cross-validation folds
        into a Json splits of indices file.

        Args:
            preprocessing_config (dict): The dict of the preprocessing main paths
            and variable values in preprocess.
            test_indices (list): The indices of the test set.
            folds (dict): The dictionary of train/val indices for each fold.

        Returns:
            None.
        """
        json_splits: dict = {}

        json_splits["test"] = test_indices
        json_splits["folds"] = folds
        
        json_splits_path: Path = Path(preprocessing_config['paths']['output_dir']) / "splits.json"
        CommonUtils.json_saving(
            json_file=json_splits,
            json_file_path=json_splits_path
        )
        print("Correctly saved indices splits Json file")
