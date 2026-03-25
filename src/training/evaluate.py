"""Standalone evaluation utilities for comparing multiple models."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from src.models.base import BaseModel


def full_report(
    model: BaseModel,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    target_names: list[str] | None = None,
) -> dict[str, Any]:
    """Return a comprehensive evaluation report for *model*.

    Parameters
    ----------
    model:
        Trained model implementing :class:`src.models.base.BaseModel`.
    X_test:
        Test feature matrix.
    y_test:
        True binary labels.
    target_names:
        Human-readable class names for the classification report.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys: ``accuracy``, ``f1``, ``roc_auc``,
        ``confusion_matrix``, ``classification_report``.
    """
    y_pred: np.ndarray = model.predict(X_test)
    y_proba: np.ndarray = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, target_names=target_names
        ),
    }
