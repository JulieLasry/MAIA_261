from pathlib import Path
from PIL import Image, ImageOps, UnidentifiedImageError
import numpy as np
import cv2 as cv
import os
import pandas as pd

from src.utils.common_utils import CommonUtils

class DataPreprocessing:
    """ 
    Loading and preprocessing of the raw data files, by resizing them and 
    enhancing the contrast with the CLAHE method. The preprocessed files 
    are then saved into a Json file, along with their label category.
    """

    @staticmethod
    def __image_preprocessing(
        image_path: Path,
        preprocessing_config: dict,
        clahe: cv.CLAHE
    ) -> Image.Image:
        """
        Load the raw image, apply a resizing and a CLAHE enhancement of
        the corresponding image.

        Args:
            image_path (Path): The path of the image to preprocess in our dataset.
            preprocessing_config (dict): The dict of the preprocessing main paths 
            and variable values in preprocess. 
            clahe (cv.CLAHE): The CLAHE object for contrast enhancement.

        Returns:
            Image.Image
            The enhanced (resized and contrast-adjusted) image.
        """
        #  Image size normalization without loosing the ratio
        with Image.open(image_path) as img:           
            padded_img: Image.Image = ImageOps.pad(
                img, 
                (preprocessing_config['data']['target_size'],
                preprocessing_config['data']['target_size'])
            )
            padded_array: np.ndarray = np.array(padded_img)

            # CLAHE image enhancement 
            enhanced_img: Image.Image = Image.fromarray(clahe.apply(padded_array))
        return enhanced_img

    @staticmethod
    def dataset_preprocessing(
        preprocessing_config: dict,
        clahe: cv.CLAHE
    ) -> None:
        """
        Apply the preprocessing steps of resizing and CLAHE
        contrast enhancement to all the files in the dataset.
        It then saves the corresponding files in the preprocessed
        folder.

        Args:
            preprocessing_config (dict): The dict of the preprocessing main paths 
            and variable values in preprocess. 
            clahe (cv.CLAHE): The CLAHE object for contrast enhancement.

        Returns:
            None.
        """
        raw_images_path: Path = Path(preprocessing_config['paths']['base_dir']) / 'check-X-ray'
        preprocessed_output_path: Path = Path(preprocessing_config['paths']['output_dir']) / 'check-X-ray'
        os.makedirs(preprocessed_output_path, exist_ok=True)

        # Checking once again if all raw files are valid and printing invalid cases
        invalid_images: list[Path] = []

        # Browsing the entire dataset of files to preprocess them while checking 
        # if they are valid
        for image_path in raw_images_path.glob('*'):
            try:
                with Image.open(image_path) as im:
                    im.verify()
            except (UnidentifiedImageError, OSError) as e:
                invalid_images.append(image_path)
                print(f"File {image_path.name} could not be opened: {e}")
                continue

            enhanced_img: Image.Image = DataPreprocessing.__image_preprocessing(
                image_path=image_path,
                preprocessing_config=preprocessing_config,
                clahe=clahe
            )
            enhanced_output = preprocessed_output_path / image_path.name
            enhanced_img.save(enhanced_output)
        print(f"Number of invalid preprocessed files : {len(invalid_images)} over 624 files")

    @staticmethod
    def dataset_saving(
        preprocessing_config: dict,
    ) -> None:
        """
        Saving the images path and labels into a Json file.

        Args:
            preprocessing_config (dict): The dict of the preprocessing main paths 
            and variable values in preprocess. 

        Returns:
            None.
        """
        preprocessed_files_path: Path = Path(preprocessing_config['paths']['output_dir']) / 'check-X-ray'
        csv_file_path: Path = Path(preprocessing_config['paths']['base_dir'])
        csv_file: pd.DataFrame = pd.read_csv(csv_file_path / "data_info.csv", header=0, index_col=0)

        # Creating a dictionary to combine filenames with their labels
        filenames_labels_dict: dict[str, int] = csv_file.set_index('filename')['label'].to_dict()

        invalid_preprocessed_images: list[Path] = []
        json_dataset: dict[str, list] = {}

        # Adding the image path and the labels to the json while checking if 
        # preprocessed images are still valid
        for image_path in preprocessed_files_path.glob('*'):
            if image_path.name in filenames_labels_dict:
                try:
                    with Image.open(image_path) as im:
                        im.verify()
                except (UnidentifiedImageError, OSError) as e:
                    invalid_preprocessed_images.append(image_path)
                    print(f"File {image_path.name} could not be opened: {e}")
                    continue

                json_dataset[image_path.name] = [
                    str(image_path), 
                    filenames_labels_dict[image_path.name]
                ]

        json_dataset_path: Path = Path(preprocessing_config['paths']['output_dir']) / "dataset.json"
        CommonUtils.json_saving(
            json_file=json_dataset,
            json_file_path=json_dataset_path
        )
        print(f"Number of invalid preprocessed files : {len(invalid_preprocessed_images)} over {csv_file.shape[0]} files")
