import torch 
from PIL import Image
import numpy as np
from torch.utils.data import Dataset

from src.data.data_augmentation import DataAugmentation

class ChestXRayDataset(Dataset):
    """
    Dataset class returning a list of preprocessed chest X-Ray
    samples (image path, label). Augmentations on the training 
    set are applied, with also final transformation (RGB convertion,
    normalization and tensor convertion) on each file of the split.

    Args:
        samples (list): List of image paths and labels pairs.
        training_config (dict): The dict of the training main paths 
        and variable values in training process. 
        is_train (bool): If the analyzed exam is part of the training
        set, to apply specific augmentations.
    """

    def __init__(
        self,
        samples: list[tuple[str, int]],
        training_config: dict,
        is_train: bool = False,
    ) -> None:
        self.samples = samples
        self.is_train = is_train

        # Data augmentations
        self.train_transform = DataAugmentation.train_transform(training_config=training_config)
        self.final_transform = DataAugmentation.final_transform(training_config=training_config)

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        return len(self.samples)

    def __getitem__(
        self, 
        idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Loads and transforms the image at the given index.

        Args:
            idx (int): Sample index in the dataset.

        Returns:
            tuple[torch.Tensor, torch.Tensor]
            The transformed image tensor and its label tensor.
        """
        image_path, label = self.samples[idx]
        with Image.open(image_path) as img:            
            if self.is_train:
                # Apply training augmentations on the training exams
                train_file: np.ndarray = self.train_transform(image=np.array(img))['image']
                tensor_image: torch.Tensor = self.final_transform(image=train_file)['image'] 
            else:
                tensor_image: torch.Tensor = self.final_transform(image=np.array(img))['image']

        tensor_label: torch.Tensor = torch.tensor(label, dtype=torch.long) 
        # To inspect the images validity  
        assert tensor_image.ndim == 3 and tensor_image.shape[0] == 3, \
            f"Unexpected image tensor shape: {tensor_image.shape}, expected (3, H, W)"
        return tensor_image, tensor_label, 
