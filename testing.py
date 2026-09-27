from pathlib import Path
from datetime import datetime
import json
import numpy as np
import torch
import lightning as L
from sklearn.utils.class_weight import compute_class_weight

from src.utils.common_utils import CommonUtils
from src.data.datamodule import ChestXRayDataModule
from src.models.model import ChestXRayModel
from src.testing.test_predictions import TestPredictions
from src.testing.test_metrics import TestMetrics
from src.testing.test_uncertainty import TestUncertainty
from src.testing.test_latent_space import TestLatentSpace


class Testing:
    """
    Evaluates the trained ChestXRayModel on the fixed test set, across
    every cross-validation fold's checkpoint. Standard classification
    and clinical metrics are computed per fold and averaged (mean/std).
    The ROC/PRC curves, threshold scenarios, MC Dropout uncertainty and
    latent space visualization are illustrated using the
    best-performing fold only (highest val_auprc_bacteria, read from
    outputs/train/train_results.json).
    """

    @staticmethod
    def _build_lightning_module_and_datamodule(
        fold_num: int,
        preprocessing_config: dict,
        training_config: dict,
        testing_config: dict,
    ) -> tuple:
        """
        Builds the DataModule for a given fold and loads its trained
        LightningModule from the corresponding saved checkpoint.

        Args:
            fold_num (int): The fold to load.
            preprocessing_config (dict): The preprocessing config dict.
            training_config (dict): The training config dict.
            testing_config (dict): The testing config dict.

        Returns:
            tuple
            The Lightning Module and the Datamodule of the corresponding fold.
        """
        model_dir: Path = Path(training_config['paths']['model_dir'])
        datamodule = ChestXRayDataModule(
            preprocessing_config=preprocessing_config,
            training_config=training_config,
            fold_num=fold_num
        )
        datamodule.setup()

        train_labels: list[int] = [label for _, label in datamodule.train_dataset.samples]
        class_weights: torch.Tensor = torch.tensor(
            compute_class_weight(
                class_weight='balanced',
                classes=np.array(testing_config['classes']['labels']),
                y=train_labels
            ),
            dtype=torch.float32
        )

        model = ChestXRayModel(training_config=training_config)
        checkpoint_path: Path = (
            model_dir / "checkpoint" / "per_fold"
            / f"{training_config['training']['name']}_fold_{fold_num}.ckpt"
        )

        lightning_module = TestPredictions.load_model_from_checkpoint(
            checkpoint_path=checkpoint_path,
            training_config=training_config,
            model=model,
            class_weights=class_weights,
            lr=training_config['hyperparameters']['lr_phase2']
        )

        return lightning_module, datamodule

    @staticmethod
    def run_step1_evaluate_all_folds(
        preprocessing_config: dict,
        training_config: dict,
        testing_config: dict,
    ) -> dict:
        """
        Evaluates every fold's checkpoint on the fixed test set,
        computing the classification report and clinical metrics
        for each fold, then averaging (mean/std) the key metrics
        across folds into a summary. Saves the result to a Json 
        file.

        Returns:
            dict
            Per-fold results and their mean/std summary.
        """
        test_metrics_dir: Path = Path(testing_config['paths']['metrics_dir'])
        fold_results: dict[int, dict] = {}

        for fold_num in range(1, preprocessing_config['split']['n_folds'] + 1):
            # Creating both datamodule and lightning instance 
            lightning_module, datamodule = Testing._build_lightning_module_and_datamodule(
                fold_num=fold_num, 
                preprocessing_config=preprocessing_config, 
                training_config=training_config, 
                testing_config=testing_config
            )

            # Retrieving the probs, preds, true labels and embeddings
            predictions: dict = TestPredictions.collect_test_predictions(
                lightning_module=lightning_module, 
                datamodule=datamodule
            )

            # Computing the classification (clf) and clinical metrics for all folds
            clf_report = TestMetrics.compute_classification_report(
                testing_config, predictions["labels"], predictions["preds"]
            )
            clinical_report = TestMetrics.compute_clinical_metrics(
                testing_config, predictions["labels"], predictions["preds"]
            )

            fold_results[fold_num] = {
                "classification_report": clf_report,
                "clinical_report": clinical_report,
            }

        # Saving a few information averaged across the folds (accuracy, macro recall and f2, bacteria recall)
        accuracies = [fold_results[f]["classification_report"]["accuracy"] for f in fold_results]
        macro_recalls = [fold_results[f]["classification_report"]["macro avg"]["recall"] for f in fold_results]
        macro_f2s = [fold_results[f]["classification_report"]["macro avg"]["f2_score"] for f in fold_results]
        bacteria_recalls = [fold_results[f]["classification_report"]["Bacteria"]["recall"] for f in fold_results]

        test_results: dict = {
            "date": datetime.now().strftime('%Y-%m-%d_%Hh%M'),
            "run_name": training_config['training']['name'],
            "folds": fold_results,
            "summary": {
                "accuracy": {"mean": round(float(np.mean(accuracies)), 2), 
                             "std": round(float(np.std(accuracies)), 2)},
                "macro_recall": {"mean": round(float(np.mean(macro_recalls)), 2), 
                                 "std": round(float(np.std(macro_recalls)), 2)},
                "macro_f2": {"mean": round(float(np.mean(macro_f2s)), 2), 
                             "std": round(float(np.std(macro_f2s)), 2)},
                "bacteria_recall": {"mean": round(float(np.mean(bacteria_recalls)), 2),
                                    "std": round(float(np.std(bacteria_recalls)), 2)}
            }
        }

        test_results: dict = CommonUtils.round_values(test_results)
        CommonUtils.create_path(test_metrics_dir)
        with open(test_metrics_dir / "test_results.json", "w", encoding="utf-8") as f:
            json.dump(test_results, f, indent=4)

        return test_results

    @staticmethod
    def run_step2_best_fold_analysis(
        preprocessing_config: dict,
        training_config: dict,
        testing_config: dict,
    ) -> None:
        """
        Identifies the best-performing fold from
        outputs/train/train_results.json (highest val_auprc_bacteria),
        then runs the ROC/PRC curves, confusion matrix, threshold
        scenarios, MC Dropout uncertainty and latent space
        visualization using that fold's checkpoint. Saves the results
        to a Json file.

        Returns:
            None
        """
        test_metrics_dir: Path = Path(testing_config['paths']['metrics_dir'])
        test_figures_dir: Path = Path(testing_config['paths']['figures_dir'])
        CommonUtils.create_path(test_metrics_dir)
        CommonUtils.create_path(test_figures_dir)

        # Finding the best fold of the training phase 
        train_dir: Path = Path(training_config['paths']['output_dir'])
        with open(train_dir / "train_results.json", "r", encoding="utf-8") as f:
            best_results = json.load(f)

        # Json keys are always strings, so we cast back to int after finding the best one
        best_fold_str: str = max(
            best_results["folds"],
            key=lambda f: best_results["folds"][f]["val_auprc_bacteria"]
        )
        best_fold: int = int(best_fold_str)

        # Creating the lighnting module and datamodule instances for this new test task 
        lightning_module, datamodule = Testing._build_lightning_module_and_datamodule(
            fold_num=best_fold, 
            preprocessing_config=preprocessing_config, 
            training_config=training_config, 
            testing_config=testing_config
        )
        predictions = TestPredictions.collect_test_predictions(
            lightning_module=lightning_module, 
            datamodule=datamodule
        )

        # Plotting the AUC and PRC curves and the confusion matrix 
        TestMetrics.plot_roc_and_prc(
            testing_config=testing_config, 
            labels=predictions["labels"], 
            probs=predictions["probs"], 
            output_path=test_figures_dir
        )

        TestMetrics.plot_confusion_matrix(
            testing_config=testing_config, 
            labels=predictions["labels"], 
            preds=predictions["preds"],
            output_path=test_figures_dir
        )

        # Computing the global-performance and cautious decision
        # threshold scenarios for the bacteria-infected class
        threshold_scenarios = TestMetrics.select_threshold_scenarios(
            testing_config=testing_config, 
            labels=predictions["labels"], 
            probs=predictions["probs"]
        )

        # Computing the probabilities uncertainties and the 2D latent space
        uncertainty_results = TestUncertainty.estimate_uncertainty(
            lightning_module=lightning_module, 
            datamodule=datamodule, 
            n_passes=testing_config['uncertainty']['n_passes']
        )

        embeddings_2d = TestLatentSpace.reduce_dimensions(predictions["embeddings"], testing_config)
        TestLatentSpace.plot_latent_space(
            embeddings_2d=embeddings_2d, 
            labels=predictions["labels"], 
            preds=predictions["preds"],
            testing_config=testing_config,
            output_path=test_figures_dir
        )

        # Save the best trained fold number, the threshold scenarios and the uncertainty estimations
        correct: np.ndarray = (predictions["labels"] == predictions["preds"])
        mean_uncertainty_correct = float(np.mean(uncertainty_results["std_probs"][correct]))
        mean_uncertainty_incorrect = float(np.mean(uncertainty_results["std_probs"][~correct]))
        best_fold_results: dict = {
            "date": datetime.now().strftime('%Y-%m-%d_%Hh%M'),
            "best_fold": best_fold,
            "threshold_scenarios": threshold_scenarios,
            "mean_uncertainty_corrects": mean_uncertainty_correct,
            "mean_uncertainty_incorrects": mean_uncertainty_incorrect,
        }
        best_fold_results = CommonUtils.round_values(best_fold_results)
        with open(test_metrics_dir / "best_fold_results.json", "w", encoding="utf-8") as f:
            json.dump(best_fold_results, f, indent=4)

    @staticmethod
    def run() -> None:
        """
        Loads the configs and runs the full test evaluation: per-fold
        metrics first, then the best fold's ROC/threshold/uncertainty/
        latent space analysis.
        """
        preprocessing_config: dict = CommonUtils.load_config('preprocessing_config.yaml')
        training_config: dict = CommonUtils.load_config('training_config.yaml')
        testing_config: dict = CommonUtils.load_config('testing_config.yaml')

        L.seed_everything(preprocessing_config['split']['seed'], workers=True) 

        Testing.run_step1_evaluate_all_folds(preprocessing_config, training_config, testing_config)
        Testing.run_step2_best_fold_analysis(preprocessing_config, training_config, testing_config)


if __name__ == "__main__":
    Testing.run()
