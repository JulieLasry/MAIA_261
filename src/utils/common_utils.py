from pathlib import Path
from typing import Any
import os
import json
import yaml

CONFIG_DIR: Path = Path(__file__).parent.parent.parent / 'configs'

class CommonUtils:
    """
    Shared utilities used across the preprocessing,
    training and testing pipelines. It loads the YAML configs, 
    creates output directories, saves Json files, and rounds nested
    numeric results for readability.
    """

    @staticmethod
    def load_config(filename: str) -> dict:
        """
        Loads the name of the YAML config file to use in the
        pipeline and returns its content.

        Args:
            filename (str): The name of the YAML config file to load.

        Returns:
            dict
            The loaded configuration dictionary.
        """
        path: Path = CONFIG_DIR / filename
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    @staticmethod
    def create_path(path: Path) -> None:
        """
        Ensures the given directory exists, creating any missing
        intermediate directories.

        Args:
            path (Path): The directory path to create.
        """
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def json_saving(
        json_file: dict, 
        json_file_path: Path
    ) -> None:
        """
        Saving the Json file into the targeted directory.

        Args:
            json_file (dict): The Json file dict.
            json_file_path (Path): The path where to save the Json file.

        Returns:
            None.
        """
        if json_file:
            with open(json_file_path, "w", encoding="utf-8") as f:
                json.dump(json_file, f, indent=4)

    @staticmethod
    def round_values(
        data: Any, 
        n_digits: int = 2
        ) -> Any:
        """
        Recursively rounds every float value found in a nested dict/list
        structure, to make Json results more readable.

        Args:
            data (Any): The dict, list, float (or any other value) to
            process. Non-numeric values are returned unchanged.
            n_digits (int): The number of decimal places to round
            floats to.

        Returns:
            Any
            The same structure, with every float rounded, and any
            other type of value left untouched.
        """
        if isinstance(data, dict):
            return {k: CommonUtils.round_values(v, n_digits) for k, v in data.items()}
        if isinstance(data, list):
            return [CommonUtils.round_values(v, n_digits) for v in data]
        if isinstance(data, float):
            return round(data, n_digits)
        return data
