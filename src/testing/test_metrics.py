import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import classification_report, fbeta_score,\
    multilabel_confusion_matrix, confusion_matrix, ConfusionMatrixDisplay, \
    precision_recall_curve, average_precision_score, \
    roc_curve, auc

class TestMetrics:
    """
    Computes and logs the full set of evaluation report. It 
    includes the classification metrics F1, F2, precision, 
    recall (per-class and macro-averaged). It likewise contains
    clinical metrics (sensititvity, specificity, PPV and NPV per
    class), the confusion matrix and the one-vs-rest ROC/PRC curves
    (AUROC and AUPRC also) per class. The accuracy is also available.
    Besides, two decision threshold scenarios in AUC for the for the
    global performance versus the cautious clinical (bacteria-infected
    samples FN lower rate) goal is implemented.
    """

    @staticmethod
    def compute_classification_report(
        testing_config: dict,
        labels: np.ndarray, 
        preds: np.ndarray
    ) -> dict:
        """
        Computes per-class + macro precision, recall,
        F1 and F2 scores.

        Args:
            testing_config (dict): The dictionary
            of the main paths and variables used in 
            testing.
            labels (np.ndarray): The true labels.
            preds (np.ndarray) : The predicted labels.

        Returns:
            dict
            The diction
            ary of the classification metrics report.
        """
        clf_report: dict = classification_report(
            y_true=labels,
            y_pred=preds,
            labels=testing_config['classes']['labels'],
            output_dict=True
        )

        # Adding the F2-score (macro and per-class)
        f2_macro = fbeta_score(
            y_true=labels,
            y_pred=preds,
            beta=2.0,
            labels=testing_config['classes']['labels'],
            average='macro'
        )

        f2_per_class = fbeta_score(
            y_true=labels,
            y_pred=preds,
            beta=2.0,
            labels=testing_config['classes']['labels'],
            average=None
        )

        for class_idx, class_name, f2_value in zip(
            testing_config['classes']['labels'],
            testing_config['classes']['label_names'],
            f2_per_class
        ):
            clf_report[str(class_idx)]['f2_score'] = f2_value
            clf_report[class_name] = clf_report.pop(str(class_idx))

        clf_report['macro avg']['f2_score'] = f2_macro

        return clf_report
        

    @staticmethod
    def compute_clinical_metrics(
        testing_config: dict,
        labels: np.ndarray,
        preds: np.ndarray,
    ) -> dict:
        """
        Computes, for each class the 
        Se, Sp, PPV and NPV from the 
        confusion matrix.

        Args:
            testing_config (dict): The dictionary
            of the main paths and variables used in 
            testing.
            labels (np.ndarray): The true labels.
            preds (np.ndarray): The predicted labels.

        Returns:
            dict
            The classification metrics report.
        """
        clinical_report: dict[str, dict[str, float]] = {}
        confusion_matrix = multilabel_confusion_matrix(
            y_true=labels, 
            y_pred=preds,
            labels=testing_config['classes']['labels']
        )

        # Computing the Se, Sp, PPV and NPV
        for idx, label in enumerate(testing_config['classes']['label_names']):
            tn, fp, fn, tp = confusion_matrix[idx].ravel()
            sensitivity: float = tp / (tp + fn)
            specificity: float = tn / (tn + fp)
            ppv: float = tp / (tp + fp)
            npv: float = tn / (tn + fn)

            clinical_report[label] = {
                "se": sensitivity,
                "sp": specificity,
                "ppv": ppv,
                "npv": npv
            }

        return clinical_report


    @staticmethod
    def plot_confusion_matrix(
        testing_config: dict,
        labels: np.ndarray,
        preds: np.ndarray,
        output_path: Path,
    ) -> None:
        """
        Plots and saves the confusion matrix

        Args:
            testing_config (dict): The dict of the main paths and
            variables used in testing.
            labels (np.ndarray): The true labels.
            preds (np.ndarray): The predicted
            output_path (Path): Path of the plotted figure (with .png).

        Returns:
            None
        """
        cf_matrix = confusion_matrix(
            y_true=labels, 
            y_pred=preds,
            labels=testing_config['classes']['labels']
        )
        matrix_display = ConfusionMatrixDisplay(
            confusion_matrix=cf_matrix,
            display_labels=testing_config['classes']['label_names'])

        matrix_display.plot()
        plt.savefig(output_path)
        plt.close(matrix_display.figure_)


    @staticmethod
    def plot_roc_and_prc(
        testing_config: dict,
        labels: np.ndarray,
        probs: np.ndarray,
        output_path: Path,
    ) -> dict:
        """
        Plots the one-vs-rest ROC and Precision for
        each class, and returns their AUROC/AUPRC 
        values.

        Args:
            testing_config (dict): The dict of the 
            variables used in testing.
            labels (np.ndarray): The true labels.
            probs (np.ndarray): The predicted class probabilities.
            output_path (Path): Path to save the plotted figures.

        Returns:
            dict
            The dictionary of the classes AUROC and AUPRC.
        """
        class_labels: list[int] = testing_config['classes']['labels']
        class_names: list[str] = testing_config['classes']['label_names']
        results: dict = {}

        # Computing the PRC/ROCcurve with AUPRC/AUROC
        # For each class
        for class_idx, class_name in zip(class_labels, class_names):
            binary_labels = (labels == class_idx).astype(int)
            class_probs = probs[:, class_idx]

            # PRC curve with AUPRC
            precision, recall, _ = precision_recall_curve(binary_labels, class_probs)
            auprc = average_precision_score(binary_labels, class_probs)

            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, marker='.', label=f'AUPRC = {auprc:.3f}')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve - {class_name}')
            plt.legend(loc='lower left')
            plt.grid(True)
            plt.savefig(output_path / f'prc_{class_name}.png')
            plt.close()

            # ROC curve with AUROC
            fpr, tpr, _ = roc_curve(binary_labels, class_probs)
            auroc = auc(fpr, tpr)

            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, marker='.', label=f'AUROC = {auroc:.3f}')
            plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # ligne "hasard"
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {class_name}')
            plt.legend(loc='lower right')
            plt.grid(True)
            plt.savefig(output_path / f'roc_{class_name}.png')
            plt.close()

            results[class_name] = {"auroc": auroc, "auprc": auprc}

        return results



    @staticmethod
    def select_threshold_scenarios(
        testing_config: dict,
        labels: np.ndarray,
        probs: np.ndarray,
    ) -> dict:
        """
       Computes and plots the one-vs-rest ROC curve for the
       bacteria-infected class on the test set, and proposes two
       decision threshold scenarios (global performance versus 
       cautious on the identified priority error).

        Args:
            testing_config (dict): The dict of the 
            variables used in testing.
            labels (np.ndarray): The true labels.
            probs (np.ndarray): The predicted labels.

        Returns:
            dict
            The disctionary of the global performance and 
            more cautious approach for two cases, to save 
            in a Json file further.
        """
        bacteria_idx: int = testing_config['classes']['bacteria_label_idx']  
        binary_labels = (labels == bacteria_idx).astype(int)
        bacteria_probs = probs[:, bacteria_idx]

        fpr, tpr, thresholds = roc_curve(binary_labels, bacteria_probs)

        # First case : Youden indice to obtain the global performance of the test
        youden_j = tpr - fpr 
        best_global_idx = np.argmax(youden_j)  
        global_threshold = thresholds[best_global_idx]

        # Second case : more cautious on the recall of the bacteria-samples
        target_recall: float = 0.95  
        valid_indices = np.where(tpr >= target_recall)[0]  
        cautious_idx = valid_indices[np.argmin(fpr[valid_indices])]
        cautious_threshold = thresholds[cautious_idx]

        return {
            "global_performance": {
                "threshold": float(global_threshold),
                "tpr": float(tpr[best_global_idx]),
                "fpr": float(fpr[best_global_idx]),
            },
            "cautious": {
                "threshold": float(cautious_threshold),
                "tpr": float(tpr[cautious_idx]),
                "fpr": float(fpr[cautious_idx]),
            },
        }
        
        
