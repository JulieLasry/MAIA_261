from pathlib import Path
from src.data.data_preprocessing import DataPreprocessing
from src.data.data_splitting import DataSplitting
from src.utils.preprocessing_utils import PreprocessingUtils
from src.utils.config_utils import ConfigUtils

class Preprocessing:
    """
    Processer main function to load our raw data,
    process and create associated Json files, one
    with the image paths and labels, and another one
    with the test and fold indexes for the cross validation.         
    """
    
    @staticmethod
    def run() -> dict:
        # Configs paths
        preprocessing_config: dict = ConfigUtils.load_config(
            'preprocessing_config.yaml'
        )

        # Step 1
        # Preprocess the data : resize and contrast enhancement
        DataPreprocessing.dataset_preprocessing(
            preprocessing_config=preprocessing_config,
            clahe=PreprocessingUtils.clahe_creation(
                preprocessing_config=preprocessing_config
            )
        )

        # Step 2
        # Save the dataset Json file (image paths and labels)
        DataPreprocessing.dataset_saving(
            preprocessing_config=preprocessing_config
        )

        # Step 3
        # Splitting the data into train/val/test set with a stratified K-Fold CV
        dataset_json_path: Path = Path(preprocessing_config['data']['output_dir']) / "dataset.json"
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
        
if __name__ == "__main__":
    Preprocessing.run()
