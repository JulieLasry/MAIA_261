import cv2 as cv
from pathlib import Path
import json

class PreprocessingUtils():
    """ 
    Utils class for the preprocessing steps.
    Here are a method for creating a CLAHE object once,
    and to create a Json file.
    """

    @staticmethod
    def clahe_creation(
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
    def json_saving(
        json_file: dict,
        json_file_path: Path
    ) -> None:
        """
        Saving the Json file into the targetted directory.

        Args:
            json_file (dict): The Json file dict. 
            json_file_path (Path): The path where to save the Json file.

        Returns:
            None.
        """
        if json_file:
            with open(json_file_path, "w", encoding="utf-8") as f:
                json.dump(json_file, f, indent=4)
 