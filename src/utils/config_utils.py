from pathlib import Path
import yaml

CONFIG_DIR: Path = Path(__file__).parent / 'configs'

class ConfigUtils:
    """
    Shared utility for loading YAML configuration files, used by
    both the preprocessing and training scripts.
    """

    @staticmethod
    def load_config(filename: str) -> dict:
        """
        Loads the name of the YAML config file 
        to use in the pipeline and returns its 
        content.

        Args:
            path (Path): The path of the YAML config file to load.

        Returns:
            dict
            The loaded configuration dictionary.
        """
        path: Path = CONFIG_DIR / filename
        with open(path, 'r') as f:
            return yaml.safe_load(f)
