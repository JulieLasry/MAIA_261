import torch
import torch.nn as nn
from src.data.datamodule import ChestXRayDataModule
from src.training.lightning_module import ChestXRayLightningModule

class TestUncertainty:
    """
    Estimates per-sample prediction uncertainty on the test set using
    Monte Carlo Dropout, while it keeps the model's Dropout layers active at
    inference and runs several stochastic forward passes per sample,
    using the variance across passes as the uncertainty estimate.
    """

    @staticmethod
    def enable_mc_dropout(
        model: nn.Module
    ) -> None:
        """
        Switches every Dropout layer of the model back to train mode
        as they are active, while the rest of the model stays in 
        eval mode.

        Args: 
            model (nn.Module): The model we switch the Dropout on.
        """
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.train()


    @staticmethod
    def estimate_uncertainty(
        lightning_module: ChestXRayLightningModule,
        datamodule: ChestXRayDataModule,
        n_passes: int,
    ) -> dict:
        """
        Runs different stochastic (probabilistic) forward 
        passes per test sample and computes the mean 
        prediction and its uncertainty (std) across passes.

        Args:
            lightning_module (ChestXRayLightningModule): The trained
            Lightning module, loaded from a checkpoint.
            datamodule (ChestXRayDataModule): The DataModule providing
            the test set to run the uncertainty estimation on.
            n_passes (int): The number of stochastic forward passes to
            run per test sample, used to estimate the prediction
            uncertainty.

        Returns:
            dict
            The dictionary of mean probabilities and uncertainty.
        """
        TestUncertainty.enable_mc_dropout(lightning_module.model)

        all_passes_probs: list = []
        probs_passes_tensor: torch.Tensor = torch.Tensor()
        with torch.no_grad():
            for _ in range(n_passes):
                all_probs: list = []
                for images, _ in datamodule.test_dataloader():
                    logits = lightning_module(images)
                    probs = torch.softmax(logits, dim=1)
                    all_probs.append(probs)
                all_passes_probs.append(torch.cat(all_probs))
            probs_passes_tensor = torch.stack(all_passes_probs)
            mean_probs = probs_passes_tensor.mean(dim=0).numpy()
            std_probs = probs_passes_tensor.std(dim=0).numpy()

        return {
            "mean_probs": mean_probs,
            "std_probs": std_probs
        }
        
        

