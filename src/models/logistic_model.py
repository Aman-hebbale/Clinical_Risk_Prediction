"""Logistic regression baseline wrapped in the BaseModel interface."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data.preprocess import build_preprocessing_pipeline
from src.models.base import BaseModel


class LogisticModel(BaseModel):
    """Logistic regression classifier with a full preprocessing pipeline.

    The pipeline applies median imputation and standard scaling to numeric
    features, and most-frequent imputation with one-hot encoding to
    categorical features, before passing the result to a logistic regression
    classifier.
    """

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        random_state: int = 42,
    ) -> None:
        """Initialise the model pipeline.

        Parameters
        ----------
        C:
            Inverse of regularisation strength. Smaller values specify
            stronger regularisation.
        max_iter:
            Maximum number of iterations for the solver to converge.
        random_state:
            Seed for reproducibility.
        """
        self._pipeline: Pipeline = Pipeline(
            [
                ("preprocessor", build_preprocessing_pipeline()),
                (
                    "clf",
                    LogisticRegression(
                        C=C,
                        max_iter=max_iter,
                        solver="lbfgs",
                        random_state=random_state,
                    ),
                ),
            ]
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LogisticModel":
        """Train the pipeline on *X* and *y*.

        Parameters
        ----------
        X:
            Training features.
        y:
            Binary labels (0 = no disease, 1 = disease).

        Returns
        -------
        LogisticModel
            Self, for method chaining.
        """
        self._pipeline.fit(X, y)
        return self

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
        return self._pipeline.predict(X)  # type: ignore[no-any-return]

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
        return self._pipeline.predict_proba(X)  # type: ignore[no-any-return]

    def save(self, path: Path) -> None:
        """Persist the model pipeline to *path* using pickle.

        Parameters
        ----------
        path:
            Destination file path. Parent directories are created if absent.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self._pipeline, f)

    @classmethod
    def load(cls, path: Path) -> "LogisticModel":
        """Load a ``LogisticModel`` from a pickled pipeline at *path*.

        Parameters
        ----------
        path:
            Source file path produced by :meth:`save`.

        Returns
        -------
        LogisticModel
            Reconstructed model instance.
        """
        instance = cls.__new__(cls)
        with path.open("rb") as f:
            instance._pipeline = pickle.load(f)  # noqa: S301
        return instance
