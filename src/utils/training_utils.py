from pathlib import Path
import os

OUTPUT_DIR: Path = Path(__file__).parent.parent.parent / 'training_outputs'
MODEL_DIR: Path = Path(__file__).parent.parent.parent / 'models'

class TrainingUtils:
    """
    Utils class for the training steps. Creates any missing
    directory needed for the training outputs (MLflow logs, YAML
    summary) and model checkpoints.
    """

    @staticmethod
    def create_path(path: Path) -> None:
        """
        Ensures the given directory exists, creating any missing
        intermediate directories.

        Args:
            path (Path): The directory path to create.
        """
        os.makedirs(path, exist_ok=True)
