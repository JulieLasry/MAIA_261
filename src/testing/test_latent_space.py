import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from umap import UMAP

class TestLatentSpace:
    """
    Reduces the test set latent embeddings to 2D and plots them,
    distinguishing correctly and incorrectly predicted samples per
    class.
    """

    @staticmethod
    def reduce_dimensions(
        embeddings: np.ndarray,
        testing_config: dict
    ) -> np.ndarray:
        """
        Projects the latent embeddings to 2D using UMAP.

        Args:
            embeddings (np.ndarray): The 1D vector of
            features in the last model layer before the
            final classification head.
            testing_config (dict): The dictionary
            of the main paths and variables used in 
            testing.

        Returns:
            np.ndarray
            The 2D projected embeddings.
        """
        reducer = UMAP(n_components=2, random_state=testing_config['latent_space']['seed']) 
        embeddings_2d = reducer.fit_transform(embeddings)
        return embeddings_2d
        

    @staticmethod
    def plot_latent_space(
        embeddings_2d: np.ndarray,
        labels: np.ndarray,
        preds: np.ndarray,
        testing_config: dict,
        output_path: Path,
    ) -> None:
        """
        Plots the 2D colored latent space by classes and pointing
        out correct versus incorrect predictions.
        Args:
            embeddings_2d (np.ndarray): The latent embeddings, reduced
            to 2D via UMAP.
            labels (np.ndarray): The true labels.
            preds (np.ndarray): The predicted labels.
            testing_config (dict): The dictionary
            of the main paths and variables used in 
            testing.
            output_path (Path): The path to save the plotted figure.
        """
        # Maps the predicted true labels 
        correct: np.ndarray = (labels == preds)  

        class_labels: list[int] = testing_config['classes']['labels']
        class_names: list[str] = testing_config['classes']['label_names']

        plt.figure(figsize=(10, 8))

        # Per-class symbol
        markers: list[str] = ['o', 's', '^']

        for (class_idx, class_name), marker in zip(zip(class_labels, class_names), markers):
            class_mask = (labels == class_idx)

            # Corectly classified samples
            mask_correct = class_mask & correct
            plt.scatter(
                embeddings_2d[mask_correct, 0],
                embeddings_2d[mask_correct, 1],
                marker=marker,
                color='green',
                label=f'{class_name} - correct',
                alpha=0.7
            )

            # Misclassified samples
            mask_incorrect = class_mask & ~correct 
            plt.scatter(
                embeddings_2d[mask_incorrect, 0],
                embeddings_2d[mask_incorrect, 1],
                marker=marker,
                color='red',
                label=f'{class_name} - incorrect',
                alpha=0.7
            )

        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.title('Latent Space Visualization')
        plt.legend()
        plt.savefig(output_path)
        plt.close()
        
