"""Shared metric computation used by both training and the Streamlit app."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_pred, y_proba, labels=None) -> dict:
    """Multi-class metrics: macro-averaged P/R/F1 and one-vs-rest macro AUC."""
    y_true = np.asarray(y_true)
    if labels is None:
        labels = np.arange(y_proba.shape[1])

    try:
        auc = roc_auc_score(y_true, y_proba, multi_class="ovr",
                            average="macro", labels=labels)
    except ValueError:
        # Happens when the uploaded file does not contain every class
        auc = float("nan")

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
