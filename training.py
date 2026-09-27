from pathlib import Path
import time
from datetime import datetime
import json
import numpy as np
import torch
import lightning as L
from sklearn.utils.class_weight import compute_class_weight
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from src.utils.common_utils import CommonUtils
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
    checkpoints into 'models/', MLflow logs under 'outputs/train/mlflow'
    and two Json summaries (best metrics and training time with number of
    trainable parameters per fold) into 'outputs/train/'.
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
        train_dir: Path = Path(training_config['paths']['output_dir'])
        model_dir: Path = Path(training_config['paths']['model_dir'])

        # Instanciating the datamodule and the model
        datamodule: ChestXRayDataModule = ChestXRayDataModule(
            preprocessing_config=preprocessing_config,
            training_config=training_config,
            fold_num=fold_num
        )
        datamodule.setup()
        model = ChestXRayModel(training_config=training_config)

        # Setting up the MLflow logger and Callbacks (Early Stopping, Metrics tracker, Ckpt)
        mlflow_path: Path = train_dir / "mlflow"
        CommonUtils.create_path(path=mlflow_path)

        run_name_prefix: str = training_config['training']['name']

        mlflow_experiment_name: str = f"{run_name_prefix}_fold_{fold_num}"
        mlflow_uri: str = f"sqlite:///{mlflow_path / f'{run_name_prefix}_fold_{fold_num}.db'}"
        run_name: str = f"{run_name_prefix}_fold_{fold_num}"

        mlflow_logger = MLFlowLogger(
            experiment_name=mlflow_experiment_name,
            tracking_uri=mlflow_uri,
            run_name=run_name
        )

        best_metric_tracker = ChestXRayBestMetricTracker()

        checkpoint_path: Path = model_dir / "checkpoint"
        CommonUtils.create_path(path=checkpoint_path)
        checkpoint_path_per_fold = checkpoint_path / "per_fold"
        CommonUtils.create_path(path=checkpoint_path_per_fold)
        checkpoint = ModelCheckpoint(
            dirpath=checkpoint_path_per_fold,
            filename=f"{run_name_prefix}_fold_{fold_num}",
            monitor="val_auprc_bacteria",
            mode="max",
            save_top_k=1,
            save_last=True, # resume and to compare last and best ckpt
            enable_version_counter=False,
            verbose=True
        )

        # Phase 1 - frozen backbone
        p1_early_stopping = EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=training_config['hyperparameters']['lr_es_patience_phase1'],
            verbose=True
        )

        p1_callbacks: list = [p1_early_stopping, best_metric_tracker, checkpoint]

        train_labels: list[int] = [label for _, label in datamodule.train_dataset.samples]
        class_weights: torch.tensor = torch.tensor(
            compute_class_weight(class_weight='balanced', classes=np.array([0,1,2]), y=train_labels),
            dtype=torch.float32
        )

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
            monitor="val_loss",
            mode="min",
            patience=training_config['hyperparameters']['lr_es_patience_phase2'],
            verbose=True
        )
        p2_callbacks: list = [p2_early_stopping, best_metric_tracker, checkpoint]

        # Phase 2 - progressive finetuning 
        model.unfreeze_last_stages(num_stages=training_config['hyperparameters']['unfreeze_stages'])
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
        and a time execution and number of model
        parameters summary into two Json files  
        in 'outputs/train/'.
        """
        train_dir: Path = Path(training_config['paths']['output_dir'])
        fold_best_results: dict[int, dict] = {}
        for fold_num in range(1, preprocessing_config['split']['n_folds'] + 1):
            fold_best_results[fold_num] = Training.run_fold(
                fold_num=fold_num,
                preprocessing_config=preprocessing_config,
                training_config=training_config
            )

        # Commputing the metrics mean and std across all folds 
        metrics_names: list[str] = [
            "val_auprc_bacteria", "val_loss", "val_recall_bacteria",
            "val_f2_bacteria", 'val_recall_macro', "val_f2_macro"
        ]
        metrics_summary: dict = {}
        for metric_name in metrics_names:
            values = [fold_best_results[fold_num][metric_name] for \
                       fold_num in fold_best_results]
            metrics_summary[metric_name] = {
                "mean": round(float(np.mean(values)), 2),
                "std": round(float(np.std(values)), 2)
            }
        best_results_dict: dict = {
            "date": datetime.now().strftime('%Y-%m-%d_%Hh%M'),
            "run_name": training_config['training']['name'],
            "metrics_summary": metrics_summary,
            "folds": {
                fold_num: {k: v for k, v in results.items() if k in metrics_names}
                for fold_num, results in fold_best_results.items()
            },
        }

        # Saving or updating the dictionnary summerizing
        # the metrics best results across all folds into a Json
        best_result_json_path: Path = train_dir / "train_results.json"
        if best_result_json_path.is_file():
            with open(best_result_json_path, "r", encoding="utf-8") as f:
                all_best_results = json.load(f)
        else:
            all_best_results = {}

        all_best_results.update(best_results_dict)
        with open(best_result_json_path, "w", encoding="utf-8") as f:
            json.dump(all_best_results, f, indent=4)

        # Saving the time and execution and number of parameters for each fold
        execution_summary: dict = {
            "date": datetime.now().strftime('%Y-%m-%d_%Hh%M'),
            "run_name": training_config['training']['name'],
            "folds": {
                fold_num: {
                    "execution_time_min": results["execution_time"],
                    "num_trainable_params": results["num_params"]
                }
                for fold_num, results in fold_best_results.items()
            }
        }

        execution_json_path: Path = train_dir / "execution_results.json"
        if execution_json_path.is_file():
            with open(execution_json_path, "r", encoding="utf-8") as f:
                execution_results = json.load(f)
        else:
            execution_results = {}

        execution_results.update(execution_summary)
        with open(execution_json_path, "w", encoding="utf-8") as f:
            json.dump(execution_results, f, indent=4)

        
    @staticmethod
    def run() -> None:
        """
        Loads the configs and runs step 1 then step 2.
        """
        preprocessing_config: dict = CommonUtils.load_config('preprocessing_config.yaml')
        training_config: dict = CommonUtils.load_config('training_config.yaml')

        L.seed_everything(preprocessing_config['split']['seed'], workers=True)

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
