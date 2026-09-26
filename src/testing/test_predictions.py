from pathlib import Path
import torch

from src.models.model import ChestXRayModel
from src.training.lightning_module import ChestXRayLightningModule
from src.data.datamodule import ChestXRayDataModule

class TestPredictions:
    """
    Loads a trained fold's best checkpoint and collects the 
    predictions once the model was tested. The predicted 
    probabilities, predicted classes, true labels and 
    latent embeddings for each test sample are then retrieved.
    These collected values are reused by the metrics, uncertainty
    and latent space scripts.
    """

    @staticmethod
    def load_model_from_checkpoint(
        checkpoint_path: Path,
        training_config: dict,
        model: ChestXRayModel,
        class_weights: torch.Tensor,
        lr: float
    ) -> ChestXRayLightningModule:
        """
        Loads a LightningModule from a previous model trained
        checkpoint. The module will then have the weights of 
        the trained model, and will be further tested.

        Args:
            checkpoint_path (Path): Path to the .ckpt file.
            training_config (dict): The dict of the training 
            main paths and variable values in training process.
            model (ChestXRayModel): A freshly-instantiated model, used as the
            architecture template to load the checkpoint's saved weights into.
            class_weights (torch.Tensor) : The weight for each class/label
            to obtain a weighted loss, since there is class imbalance.
            lr (float): The learning rate to use for the current phase.

        Returns:
            ChestXRayLightningModule
            The loaded Lightning module, in eval mode.
        """

        lightning_module = ChestXRayLightningModule.load_from_checkpoint(
            checkpoint_path,
            model=model,
            training_config=training_config,
            class_weights=class_weights,
            lr=lr
        )
        lightning_module.eval()
        return lightning_module


    @staticmethod
    def collect_test_predictions(
        lightning_module: ChestXRayLightningModule,
        datamodule: ChestXRayDataModule,
    ) -> dict:
        """
        Test the lightning module loaded with the checkpoint, 
        and collects the predicted probabilities and labels, 
        with also the true labels and latent embedding for 
        further analysis, such as the latent space.

        Args:
            lightning_module (ChestXRayLightningModule): The 
            lightning module loaded with the checkpoint and tested.
            datamodule (ChestXRayDataModule): The datamodule where
            we use its test samples.

        Returns:
            dict
            Dictionnary of the probabilities, predictions, labels
            and embeddings. 
        """
        all_probs, all_preds, all_labels, all_embeddings = [], [], [], []

        with torch.no_grad():
            for images, labels in datamodule.test_dataloader():
                logits = lightning_module(images)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                embeddings = lightning_module.model.return_embedding(images)

                all_probs.append(probs)
                all_preds.append(preds)
                all_labels.append(labels)
                all_embeddings.append(embeddings)

        return {
            "probs": torch.cat(all_probs).numpy(),
            "preds": torch.cat(all_preds).numpy(),
            "labels": torch.cat(all_labels).numpy(),
            "embeddings": torch.cat(all_embeddings).numpy(),
        }
