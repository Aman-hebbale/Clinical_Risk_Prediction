"""Random forest baseline wrapped in the BaseModel interface."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.base import BaseModel


class RandomForestModel(BaseModel):
    """Random forest classifier with a standard-scaling preprocessing step.

    Uses ``StandardScaler`` for feature normalisation before fitting a
    ``RandomForestClassifier``.  Tree-based ensembles are not sensitive to
    feature scale, but including the scaler keeps the interface consistent
    with other baseline pipelines and makes the step explicit.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int | None = None,
        random_state: int = 42,
    ) -> None:
        """Initialise the model pipeline.

        Parameters
        ----------
        n_estimators:
            Number of trees in the forest.
        max_depth:
            Maximum depth of each tree.  ``None`` expands nodes until all
            leaves are pure or contain fewer than ``min_samples_split``
            samples.
        random_state:
            Seed for reproducibility.
        """
        self._pipeline: Pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomForestModel":
        """Train the pipeline on *X* and *y*.

        Parameters
        ----------
        X:
            Training features.
        y:
            Binary labels (0 = no disease, 1 = disease).

        Returns
        -------
        RandomForestModel
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
    def load(cls, path: Path) -> "RandomForestModel":
        """Load a ``RandomForestModel`` from a pickled pipeline at *path*.

        Parameters
        ----------
        path:
            Source file path produced by :meth:`save`.

        Returns
        -------
        RandomForestModel
            Reconstructed model instance.
        """
        instance = cls.__new__(cls)
        with path.open("rb") as f:
            instance._pipeline = pickle.load(f)  # noqa: S301
        return instance
