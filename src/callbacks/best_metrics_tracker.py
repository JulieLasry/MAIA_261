import torch
import lightning as L

class ChestXRayBestMetricTracker(L.Callback):
    """
    Tracks the best validation AUPRC (bacteria class) seen across
    epochs, and saves the other validation metrics logged at
    that same epoch, for a final per-fold reporting.
    """  
    def __init__(
        self
    ) -> None:
        self.val_auprc_bacteria: float = float("-inf")
        self.val_loss_at_best: float = float("nan")
        self.val_recall_macro_at_best: float = float("nan")
        self.val_f2_macro_at_best: float = float("nan")
        self.val_recall_bacteria_at_best: float = float("nan")
        self.val_f2_bacteria_at_best: float = float("nan")
        
    def on_validation_epoch_end(
        self,
        trainer: L.Trainer,
        l_module: L.LightningModule
    ) -> None:
        """
        Checks if this epoch improved the tracked best AUPRC
        (bacteria class), and if so, saves the other metrics.
        """
        metrics = trainer.callback_metrics
        val_auprc_bacteria = metrics.get("val_auprc_bacteria")

        if val_auprc_bacteria is not None and not torch.isnan(val_auprc_bacteria) \
              and float(val_auprc_bacteria) > self.val_auprc_bacteria:
            # Saving the best monitored metric and other informative ones
            self.val_auprc_bacteria = float(val_auprc_bacteria)
            self.val_loss_at_best = float(metrics.get("val_loss", float("nan")))
            self.val_recall_macro_at_best = float(metrics.get("val_recall_macro", float("nan")))
            self.val_f2_macro_at_best = float(metrics.get("val_f2_macro", float("nan")))
            self.val_recall_bacteria_at_best = float(metrics.get("val_recall_bacteria", float("nan")))
            self.val_f2_bacteria_at_best = float(metrics.get("val_f2_bacteria", float("nan")))
