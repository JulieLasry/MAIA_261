import albumentations as A
from albumentations.pytorch import ToTensorV2

class DataAugmentation():
    """
    Create the transformations for all files and
    the ones specific to the train set. It includes
    geometric and luminosity transformations using
    the Albumentations library. 
    """
    
    @staticmethod
    def train_transform(
        training_config: dict,
    ) -> A.Compose :
        """
        Transform only the train files. Here, geometric 
        and photmetric transformations are applied.

        Args:
            training_config (dict): The dict of the training main paths 
            and variable values in training process. 

        Returns:
            A.Compose
            The composed tranformation for the training files.
        """
        return A.Compose([
            A.HorizontalFlip(p=training_config['augmentation']['horizontal_flip']), 
            A.Affine(
                scale=training_config['augmentation']['scaling'], 
                translate_percent=training_config['augmentation']['translation'],
                rotate=training_config['augmentation']['rotation'],
                shear=training_config['augmentation']['shear']
            ),
            A.GaussNoise(std_range=training_config['augmentation']['gaussian_noise']),
            A.RandomBrightnessContrast(
                brightness_limit=training_config['augmentation']['brightness'],
                contrast_limit=training_config['augmentation']['contrast']
            )
        ])

    @staticmethod
    def final_transform(
        training_config: dict,
    ) -> A.Compose :
        """
        Transform already preprocessed val and test files,
        with also already augmented train files. Images are here
        converted to RGB, for the sake of Transfer Learning with 
        ImageNet, then normalized and converted into tensors.

        Args:
            preprocessing_config (dict): Base config dictionnary defining the 
                main paths, tasks and parameter values for the preprocessing task.

        Returns:
            A.Compose
            The final transformed composed object.
        """
        return A.Compose([
            A.ToRGB(),
            A.Normalize(
                mean=training_config['normalization']['mean'],
                std=training_config['normalization']['std'],
                max_pixel_value=training_config['normalization']['max_pixel_value']
            ),
            ToTensorV2()
        ])
