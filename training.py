from pathlib import Path
import time
import json
import numpy as np
import torch
import lightning as L
from sklearn.utils.class_weight import compute_class_weight
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from src.utils.config_utils import ConfigUtils
from src.utils.training_utils import TrainingUtils, OUTPUT_DIR, MODEL_DIR

from src.data.datamodule import ChestXRayDataModule
from src.models.model import ChestXRayModel
from src.training.lightning_module import ChestXRayLightningModule
from src.callbacks.best_metrics_tracker import ChestXRayBestMetricTracker


class Training:
    """
    Trains the ChestXRayModel across each stratified cross-validation
    fold in two phases (frozen feature extractor, then progressive
    fine-tuning). Step 1 validates the data/split by printing the
    dataloader lengths per fold. Step 2 trains each fold, saving
    checkpoints into 'models/', MLflow logs under 'training_outputs/mlflow'
    and a YAML summary (best metrics, training time, number of
    trainable parameters per fold) into 'training_outputs/training_summary.yaml'.
    """

    @staticmethod
    def run_step1_print_dataloader_lengths(
        preprocessing_config: dict,
        training_config: dict,
    ) -> None:
        """
        Builds the DataModule for each fold and prints the length
        of its train/val/test datasets, to validate the length of 
        the available data and splits before training.
        """
        for fold_num in range(1, preprocessing_config['split']['n_folds']+1):
            datamodule: ChestXRayDataModule = ChestXRayDataModule(
                preprocessing_config=preprocessing_config,
                training_config=training_config,
                fold_num=fold_num
            )
            datamodule.setup()
            for split in ("train", "val", "test"):
                dataset = getattr(datamodule, f"{split}_dataset")
                if dataset is not None:
                    print(f"Fold : {fold_num} - {split} length : {len(dataset)}")


    @staticmethod
    def run_fold(
        fold_num: int,
        preprocessing_config: dict, 
        training_config: dict,
    ) -> dict:
        """
        Trains the model for a single fold, accross both phases
        measuring execution time and the number of trainable
        parameters.

        Returns:
            dict
            This fold's summary (best metrics, training time,
            parameter count).
        """
        start_time: float = time.perf_counter()

        # Instanciating the datamodule and the model
        datamodule: ChestXRayDataModule = ChestXRayDataModule(
            preprocessing_config=preprocessing_config,
            training_config=training_config,
            fold_num=fold_num
        )
        datamodule.setup()
        model = ChestXRayModel(training_config=training_config)

        # Setting up the MLflow logger and Callbacks (Early Stopping, Metrics tracker, Ckpt)
        mlflow_path: Path = OUTPUT_DIR / "mlflow"
        TrainingUtils.create_path(path=mlflow_path)

        mlflow_experiment_name: str = f"fold_{fold_num}"
        mlflow_uri: str = f"sqlite:///{mlflow_path / f'fold_{fold_num}.db'}"
        run_name: str = f"fold_{fold_num}"

        mlflow_logger = MLFlowLogger(
            experiment_name=mlflow_experiment_name,
            tracking_uri=mlflow_uri,
            run_name=run_name
        )

        callbacks: list = []
        p1_early_stopping = EarlyStopping(
            monitor="val_auprc_bacteria",
            mode="max",
            patience=training_config['hyperparameters']['lr_es_patience'],
            verbose=True
        )

        best_metric_tracker = ChestXRayBestMetricTracker()

        checkpoint_path: Path = MODEL_DIR / "checkpoint"
        TrainingUtils.create_path(path=checkpoint_path)
        checkpoint = ModelCheckpoint(
            dirpath=checkpoint_path,
            filename=f"fold_{fold_num}",
            monitor="val_auprc_bacteria",
            mode="max",
            save_top_k=1,
            save_last=True, # resume and to compare last and best ckpt
            verbose=True
        )
        p1_callbacks: list = [p1_early_stopping, best_metric_tracker, checkpoint]

        # Creating the Lightning module and the Trainer
        train_labels: list[int] = [label for _, label in datamodule.train_dataset.samples]
        class_weights: torch.tensor = torch.tensor(
            compute_class_weight(class_weight='balanced', classes=np.array([0,1,2]), y=train_labels),
            dtype=torch.float32
        )

        # Phase 1 - frozen backbone
        p1_ligntning_module = ChestXRayLightningModule(
            model=model,
            training_config=training_config,
            class_weights=class_weights,
            lr=training_config['hyperparameters']['lr_phase1'],
        )
        p1_trainer: L.Trainer = L.Trainer(
            max_epochs=training_config['hyperparameters']['max_epochs_phase1'],
            accelerator="auto",
            devices="auto",
            callbacks=p1_callbacks,
            deterministic=True,
            precision="16-mixed" if training_config['training']['amp'] else "32",
            logger=mlflow_logger
        )
        p1_trainer.fit(p1_ligntning_module, datamodule=datamodule)

        #  Reinitializing the Early Stopping for phase 2
        p2_early_stopping = EarlyStopping(
            monitor="val_auprc_bacteria",
            mode="max",
            patience=training_config['hyperparameters']['lr_es_patience'],
            verbose=True
        )
        p2_callbacks: list = [p2_early_stopping, best_metric_tracker, checkpoint]

        # Phase 2 - progressive finetuning 
        model.unfreeze_last_stages(num_stages=2)
        p2_ligntning_module = ChestXRayLightningModule(
            model=model,
            training_config=training_config,
            class_weights=class_weights,
            lr=training_config['hyperparameters']['lr_phase2'],
        )
        p2_trainer: L.Trainer = L.Trainer(
            max_epochs=training_config['hyperparameters']['max_epochs_phase2'],
            accelerator="auto",
            devices="auto",
            callbacks=p2_callbacks,
            deterministic=True,
            precision="16-mixed" if training_config['training']['amp'] else "32",
            logger=mlflow_logger
        )
        p2_trainer.fit(p2_ligntning_module, datamodule=datamodule)

        elapsed_min: float = float(time.perf_counter() - start_time) / 60
        num_params: int = sum(p.numel() for p in model.parameters() if p.requires_grad)

        return {
            "val_auprc_bacteria": best_metric_tracker.val_auprc_bacteria,
            "val_loss": best_metric_tracker.val_loss_at_best,
            "val_recall_bacteria": best_metric_tracker.val_recall_bacteria_at_best,
            "val_f2_bacteria": best_metric_tracker.val_f2_bacteria_at_best,
            "val_recall_macro": best_metric_tracker.val_recall_macro_at_best,
            "val_f2_macro": best_metric_tracker.val_f2_macro_at_best,
            "execution_time": round(elapsed_min, 2),
            "num_params": num_params
        }


    @staticmethod
    def run_step2_train_all_folds(
        preprocessing_config: dict,
        training_config: dict,
    ) -> None:
        """
        Runs the run_fold() method for every fold, 
        then saves the combined summary of all folds 
        to a Json file in 'training_outpus/'.
        """
        fold_best_results: dict[int, dict] = {}
        for fold_num in range(1, preprocessing_config['n_folds']+1):
            fold_best_results[fold_num] = Training.run_fold(
                fold_num=fold_num,
                preprocessing_config=preprocessing_config,
                training_config=training_config
            )

        # Saving the metrics mean and std across all folds 
        metrics_names: list[str] = [
            "val_auprc_bacteria", "val_loss", "val_recall_bacteria",
            "val_f2_bacteria", 'val_recall_macro', "val_f2_macro"
        ]
        best_results_dict: dict = {"folds": best_results_dict, "mean": {}}
        for metric_name in metrics_names:
            values = [best_results_dict[fold_num][metric_name] for fold_num in best_results_dict]
            best_results_dict["mean"][metric_name] = round(float(np.mean(values)), 2)
            best_results_dict["std"][f"{metric_name}_std"] = round(float(np.std(values)), 2)

        best_result_json_path: Path = OUTPUT_DIR / "best_results.json"
        if best_result_json_path.is_file():
            with open(best_result_json_path, "r", encoding="utf-8") as f:
                all_best_results = json.load(f)
        else:
            all_best_results = {}

        # Updating the dictionary with the new values
        all_best_results.update(best_results_dict)
        with open(best_result_json_path, "w", encoding="utf-8") as f:
            json.dump(all_best_results, f, indent=4)
        

    @staticmethod
    def run() -> None:
        """
        Loads the configs and runs step 1 then step 2.
        """
        preprocessing_config: dict = ConfigUtils.load_config('preprocessing_config.yaml')
        training_config: dict = ConfigUtils.load_config('training_config.yaml')

        Training.run_step1_print_dataloader_lengths(
            preprocessing_config=preprocessing_config, 
            training_config=training_config
        )
        Training.run_step2_train_all_folds(
            preprocessing_config=preprocessing_config,
            training_config=training_config
        )

if __name__ == "__main__":
    Training.run()
