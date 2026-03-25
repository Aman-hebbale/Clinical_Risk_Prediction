"""Abstract base class for all MedPredict models."""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd


class BaseModel(ABC):
    """Common interface that every model implementation must satisfy."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseModel":
        """Train the model on *X* and *y*.

        Parameters
        ----------
        X:
            Training features.
        y:
            Binary labels (0 = no disease, 1 = disease).

        Returns
        -------
        BaseModel
            Self, for method chaining.
        """
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return binary class predictions.

        Parameters
        ----------
        X:
            Feature matrix.

        Returns
        -------
        np.ndarray
            Integer array of shape (n_samples,).
        """
        ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return class probabilities.

        Parameters
        ----------
        X:
            Feature matrix.

        Returns
        -------
        np.ndarray
            Float array of shape (n_samples, 2).
        """
        ...

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist the model to *path*."""
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BaseModel":
        """Load a model from *path*."""
        ...
