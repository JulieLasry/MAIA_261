import yaml
from pathlib import Path
import cv2 as cv
from src.data.data_preprocessing import DataPreprocessing
from src.utils.preprocessing_utils import PreprocessingUtils

CONFIG_DIR: Path = Path(__file__).parent / 'configs'

class Preprocessing:
    """
    Processer main function to load our raw data,
    process and create associated Json files, one
    with the image paths and labels, and another one
    with the test and fold indexes for the cross validation.         
    """

    @staticmethod
    def load_config(path: Path) -> dict:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    @staticmethod
    def run() -> dict:
        # Configs paths
        preprocessing_config: dict = Preprocessing.load_config(
            CONFIG_DIR / 'preprocessing_config.yaml'
        )

        # Preprocess the data : resize and contrast enhancement
        DataPreprocessing.dataset_preprocessing(
            preprocessing_config=preprocessing_config,
            clahe=PreprocessingUtils.clahe_creation(
                preprocessing_config=preprocessing_config
            )
        )

        # Save the dataset Json file (image paths and labels)
        DataPreprocessing.dataset_saving(
            preprocessing_config=preprocessing_config
        )
        
if __name__ == "__main__":
    Preprocessing.run()
