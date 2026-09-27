import lightning as L
import torch
import torch.nn as nn
from typing import Optional
from torchmetrics import Recall, FBetaScore, AveragePrecision
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig


class ChestXRayLightningModule(L.LightningModule):
    """
    Lightning module for training the ChestXRayModel. It defines
    the training, validation and test steps and the optimizer,
    using the learning rate set for the current training phase
    (feature extractor or progressive finetuning of the backbone).
    For validation, the AUPRC of the bacteria-infected class is
    logged (used for checkpoint selection), along with recall and
    F2 score for both the bacteria-infected class and macro-averaged
    across all classes.

    Args:
        model (nn.Module): The classification model (ChestXRayModel).
        training_config (dict): The dict of the training main paths
        and variable values in training process.
        class_weights (torch.Tensor) : The weight for each class/label
        to obtain a weighted loss, since there is class imbalance.
        lr (float): The learning rate to use for the current phase.
    """

    def __init__(
        self,
        model: nn.Module,
        training_config: dict,
        class_weights: torch.Tensor,
        lr: float,
    ) -> None:
        super().__init__()
        self.model = model
        self.training_config = training_config
        self.lr = lr
        self.bacteria_label_idx = self.training_config['metrics']['bacteria_label_idx']

        # Loss 
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Metrics
        # For the Checkpoint
        self.auprc = AveragePrecision(
            task="multiclass", 
            num_classes=self.training_config['model']['num_classes'], 
            average=None
        ) 

        # To track the overall model performance, despite class imbalance
        self.recall_macro = Recall(
            task="multiclass", 
            num_classes=self.training_config['model']['num_classes'], 
            average="macro"
        )
        self.f2_macro = FBetaScore(
            task="multiclass", 
            num_classes=self.training_config['model']['num_classes'], 
            beta=2.0, 
            average="macro"
        )   

        # To track the bacteria-infected samples classification
        self.recall = Recall(
            task="multiclass", 
            num_classes=self.training_config['model']['num_classes'], 
            average=None
        ) 
        self.f2 = FBetaScore(
            task="multiclass", 
            num_classes=self.training_config['model']['num_classes'], 
            beta=2.0, 
            average=None
        ) 

    def forward(
        self, 
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward prediction pass, where
        images inputs feed the model.

        Returns:
            torch.Tensor
            The raw logits for a batch of images
        """
        return self.model(x)

    def training_step(
        self, 
        batch: Optional[tuple[torch.Tensor, Optional[torch.Tensor]]], 
        batch_idx: int
    ) -> torch.Tensor:
        """
        Computes the weighted cross-entropy loss on a training
        batch, logs it, and returns it for backpropagation.

        Args:
            batch: Loaded training batch of the image and label tensors.
            batch_idx: Batch index (unused).
        
        Returns:
            torch.Tensor
            The training Cross-Entropy loss.
        """
        images, labels = batch
        preds = self(images)

        # Computes, logs  and returns the training loss for backpropagation
        training_loss: torch.Tensor = self.criterion(preds, labels)
        self.log("train_loss", training_loss, on_step=True, on_epoch=True, prog_bar=True,
            batch_size=preds.shape[0])

        return training_loss
            
    def validation_step(
        self, 
        batch: Optional[tuple[torch.Tensor, Optional[torch.Tensor]]], 
        batch_idx: int
    ) -> None:
        """
        Computes the validation loss and updates the validation
        metrics (AUPRC (bacteria-infected samples), recall and F2
        (both macro-averaged and for bacteria-infected samples)) 
        for this batch.

        Args:
            batch: Loaded validation batch of the image and label tensors.
            batch_idx: Batch index (unused).
        """
        images, labels = batch
        logits = self(images)
        probs = torch.softmax(logits, dim=1)

        # Computes the validation loss and logs it
        val_loss: torch.Tensor = self.criterion(logits, labels)
        self.log("val_loss", val_loss, on_step=True, on_epoch=True, prog_bar=True,
            batch_size=logits.shape[0])

        # Updates the validation metrics (computed in best_metric_tracker)
        self.auprc.update(probs, labels)
        self.recall_macro.update(probs, labels)
        self.f2_macro.update(probs, labels)
        self.recall.update(probs, labels)
        self.f2.update(probs, labels)

    def on_validation_epoch_end(
        self
    ) -> None:
        """
        Computes, logs and resets the validation metrics
        at the end of each epoch. 
        """
        auprc_per_class: float = self.auprc.compute()
        recall_per_class: float = self.recall.compute()
        f2_per_class: float = self.f2.compute()

        self.log("val_auprc_bacteria", auprc_per_class[self.bacteria_label_idx])
        self.log("val_recall_bacteria", recall_per_class[self.bacteria_label_idx])
        self.log("val_f2_bacteria", f2_per_class[self.bacteria_label_idx])
        self.log("val_recall_macro", self.recall_macro.compute())
        self.log("val_f2_macro", self.f2_macro.compute())

        for metric in (self.auprc, self.recall_macro, self.f2_macro, self.recall, self.f2):
            metric.reset()

    def configure_optimizers(
        self
        ) -> OptimizerLRSchedulerConfig:
        """
        Builds the optimizer from the config file, on
        currently trainable parameters, using the available 
        LR for the training phase.

        Returns:
            The configuration of the learning rate scheduler.
        """
        optimizer_class = getattr(torch.optim, self.training_config['hyperparameters']['optimizer'])
        optimizer = optimizer_class(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.lr,
            weight_decay=self.training_config['hyperparameters']['weight_decay']
        )

        # Scheduler to reduce by half the LR value if no improvement is seen on the val loss
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=self.training_config['hyperparameters']['lr_scheduler_patience'],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
            },
        }
