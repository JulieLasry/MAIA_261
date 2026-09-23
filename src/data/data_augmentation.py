import albumentations as A
from albumentations.pytorch import ToTensorV2

class Transform():
    """
    Create the transformations for all files and
    the ones specific to the train set. It includes
    geometric and luminosity transformations using
    the Albumentations library. 
    """
    
    @staticmethod
    def base_transform(
        preprocessing_config: dict,
    ) -> A.Compose :
        """
        Transform all train/val/test raw files. 
        Resizing, additional padding and RGB 
        conversion is applied. This compose 
        transformation is the first applied to 
        the files.

        Args:
            preprocessing_config (dict): Base config dictionnary defining the 
                main paths, tasks and parameter values for the preprocessing task.

        Returns:
            The base transformed composed object, created with
            the Albumentations library.
        """
        return A.Compose([
            A.LongestMaxSize(max_size=preprocessing_config['data']['img_size']),
            A.PadIfNeeded(
                min_height=preprocessing_config['data']['img_size'], 
                min_width=preprocessing_config['data']['img_size'], 
                border_mode=0),
            A.ToRGB()
        ])


    @staticmethod
    def train_transform(
        preprocessing_config: dict,
    ) -> A.Compose :
        """
        Transform only the train files.
        Geometric and luminosity transformations
        are applied.

        Args:
            preprocessing_config (dict): Base config dictionnary defining the 
                main paths, tasks and parameter values for the preprocessing task.

        Returns:
            The train transformed composed object, created with
            the Albumentations library.
        """
        return A.Compose([
            A.HorizontalFlip(p=preprocessing_config['augmentation']['horizontal_flip']), 
            A.Affine(
                scale=preprocessing_config['augmentation']['scaling'], 
                translate_percent=preprocessing_config['augmentation']['translation'],
                rotate=preprocessing_config['augmentation']['rotation'],
                shear=preprocessing_config['augmentation']['shear']
            ),
            A.GaussNoise(std_range=preprocessing_config['augmentation']['gaussian_noise']),
            A.RandomBrightnessContrast(
                brightness_limit=preprocessing_config['augmentation']['brightness'],
                contrast_limit=preprocessing_config['augmentation']['contrast']
            )
        ])

    @staticmethod
    def final_transform(
        preprocessing_config: dict,
    ) -> A.Compose :
        """
        Transform already processed 
        train/val/test files. 
        Normalization and tensor conversion
        are applied.

        Args:
            preprocessing_config (dict): Base config dictionnary defining the 
                main paths, tasks and parameter values for the preprocessing task.

        Returns:
            The final transformed composed object, created with
            the Albumentations library.
        """
        return A.Compose([
            A.Normalize(
                mean=preprocessing_config['normalization']['mean'],
                std=preprocessing_config['normalization']['std'],
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ])
