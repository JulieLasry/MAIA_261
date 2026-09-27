import cv2 as cv
from pathlib import Path
from src.data.data_preprocessing import DataPreprocessing
from src.data.data_splitting import DataSplitting
from src.utils.common_utils import CommonUtils

class Preprocessing:
    """
    Runs the preprocessing 3-steps pipeline. It resizes and enhances 
    the raw images, saves the dataset Json file (image paths and labels),
    then splits the data into a stratified test set and stratified
    K-Fold cross-validation folds.
    """

    @staticmethod
    def _clahe_creation(
        preprocessing_config: dict
    ) -> cv.CLAHE:
        """
        Creating a CLAHE (Contrast Limited Adaptive Histogram Equalization)
        object, to be reused across all images.

        Args:
            preprocessing_config (dict): The dict of the preprocessing main paths 
            and variable values in preprocess. 

        Returns:
            cv.CLAHE
            The configured CLAHE object.
        """
        return cv.createCLAHE(
            clipLimit=preprocessing_config['data']['clipLimit'], 
            tileGridSize=(
                preprocessing_config['data']['tileGridSize'], 
                preprocessing_config['data']['tileGridSize']
            )
        )

    @staticmethod
    def run_step1_preprocess_images(preprocessing_config: dict) -> None:
        """
        Resizes and applies CLAHE contrast enhancement to every raw
        image, saving the results to the preprocessed output folder.

        Args:
            preprocessing_config (dict): The dict of the preprocessing 
            main paths and variable values in preprocess. 

        Returns:
            None
        """
        DataPreprocessing.dataset_preprocessing(
            preprocessing_config=preprocessing_config,
            clahe=Preprocessing._clahe_creation(
                preprocessing_config=preprocessing_config
            )
        )

    @staticmethod
    def run_step2_save_dataset_json(preprocessing_config: dict) -> None:
        """
        Matches each preprocessed image to its label and saves the
        paths/labels mapping into the dataset Json file.

        Args:
            preprocessing_config (dict): The dict of the preprocessing 
            main paths and variable values in preprocess. 

        Returns:
            None
        """
        DataPreprocessing.dataset_saving(
            preprocessing_config=preprocessing_config
        )

    @staticmethod
    def run_step3_split_dataset(preprocessing_config: dict) -> None:
        """
        Splits the preprocessed dataset into a stratified test set
        and stratified K-Fold cross-validation folds, then saves the
        resulting indices to the splits Json file.

        Args:
            preprocessing_config (dict): The dict of the preprocessing 
            main paths and variable values in preprocess. 

        Returns:
            None
        """
        dataset_json_path: Path = Path(preprocessing_config['paths']['output_dir']) / "dataset.json"
        try:
            train_val_indices, test_indices, labels = DataSplitting.split_train_test(
                dataset_json_path=dataset_json_path,
                preprocessing_config=preprocessing_config
            )

            folds = DataSplitting.create_cv_folds(
                preprocessing_config=preprocessing_config,
                train_val_indices=train_val_indices,
                labels=labels
            )

            DataSplitting.save_splits(
                preprocessing_config=preprocessing_config,
                test_indices=test_indices,
                folds=folds
            )
        except FileNotFoundError as e:
            print(f"dataset.json file not found. Check if the previous preprocessing steps succeeded : {e}")

    @staticmethod
    def run() -> None:
        """
        Loads the preprocessing config and runs the 3 preprocessing
        steps in order.

        Returns:
            None
        """
        preprocessing_config: dict = CommonUtils.load_config('preprocessing_config.yaml')

        Preprocessing.run_step1_preprocess_images(preprocessing_config=preprocessing_config)
        Preprocessing.run_step2_save_dataset_json(preprocessing_config=preprocessing_config)
        Preprocessing.run_step3_split_dataset(preprocessing_config=preprocessing_config)


if __name__ == "__main__":
    Preprocessing.run()
