from pathlib import Path
from PIL import Image, ImageOps, UnidentifiedImageError
import numpy as np
import cv2 as cv
import os
import pandas as pd
import json

class DataSplitting():
    """ 
    Loading and preprocessing of the raw data files, by resizing them and 
    enhancing the contrast with the CLAHE method. The preprocessed files 
    are then saved into a Json file, along with their label category.
    """

    @staticmethod
    def __image_preprocessing(
        image_path: Path,
        preprocessing_config: dict
    ) -> Image.Image:
        """
        Load the raw image, apply a resizing and a CLAHE enhancement of
        the corresponding image.

        Args:
            image_path (Path): The path of the image to preprocess in our dataset.
            preprocessing_config (dict): The dict of the preprocessing main paths 
            and variable values in preprocess. 

        Returns:
            Image.Image
            The enhanced (resized and contrast-adjusted) image.
        """
