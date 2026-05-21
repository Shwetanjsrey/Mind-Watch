from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    cohen_kappa_score,
    mean_absolute_error
)
import numpy as np


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="macro"
    )

    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="weighted"
    )

    acc = accuracy_score(labels, predictions)

    qwk = cohen_kappa_score(labels, predictions, weights="quadratic")

    mae = mean_absolute_error(labels, predictions)

    return {
        "accuracy": acc,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "weighted_f1": weighted_f1,
        "qwk": qwk,
        "mae": mae
    }


def get_confusion(labels, preds):
    return confusion_matrix(labels, preds)