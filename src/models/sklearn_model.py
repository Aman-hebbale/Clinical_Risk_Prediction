"""scikit-learn gradient boosting model wrapped in the BaseModel interface."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.base import BaseModel


class SklearnModel(BaseModel):
    """Gradient boosting classifier with an integrated preprocessing pipeline."""

    def __init__(
        self,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        max_depth: int = 3,
        random_state: int = 42,
    ) -> None:
        """Initialise the gradient boosting pipeline.

        Parameters
        ----------
        n_estimators:
            Number of boosting stages to run.
        learning_rate:
            Shrinkage factor applied to each tree's contribution.
        max_depth:
            Maximum depth of each individual regression estimator.
        random_state:
            Seed for reproducibility.
        """
        self._pipeline: Pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    GradientBoostingClassifier(
                        n_estimators=n_estimators,
                        learning_rate=learning_rate,
                        max_depth=max_depth,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SklearnModel":
        """Train the pipeline on *X* and *y*.

        Parameters
        ----------
        X:
            Training features.
        y:
            Binary labels (0 = no disease, 1 = disease).

        Returns
        -------
        SklearnModel
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
    def load(cls, path: Path) -> "SklearnModel":
        """Load a ``SklearnModel`` from a pickled pipeline at *path*.

        Parameters
        ----------
        path:
            Source file path produced by :meth:`save`.

        Returns
        -------
        SklearnModel
            Reconstructed model instance.
        """
        instance = cls.__new__(cls)
        with path.open("rb") as f:
            instance._pipeline = pickle.load(f)  # noqa: S301
        return instance
