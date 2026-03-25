"""Preprocessing utilities: cleaning, encoding, and train/test splitting."""

import logging

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger: logging.Logger = logging.getLogger(__name__)

RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2

# Features with missing values that require imputation
NUMERIC_FEATURES: list[str] = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
]
CATEGORICAL_FEATURES: list[str] = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
]


def binarise_target(y: pd.Series) -> pd.Series:
    """Convert multi-class target (0-4) to binary (0 = no disease, 1 = disease).

    Parameters
    ----------
    y:
        Raw target series from the UCI dataset.

    Returns
    -------
    pd.Series
        Binary target series.
    """
    return (y > 0).astype(np.int8)


def drop_missing(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Drop rows where any feature value is missing.

    Parameters
    ----------
    X:
        Feature matrix.
    y:
        Target series aligned with *X*.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        Cleaned (X, y) pair.
    """
    mask: pd.Series = X.notna().all(axis=1)
    n_dropped: int = int((~mask).sum())
    if n_dropped:
        logger.warning("Dropping %d rows with missing values", n_dropped)
    return X[mask].reset_index(drop=True), y[mask].reset_index(drop=True)


def build_numeric_pipeline() -> Pipeline:
    """Return a sklearn Pipeline that imputes then scales numeric features.

    Returns
    -------
    Pipeline
        Two-step pipeline: median imputation followed by standard scaling.
    """
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def build_categorical_pipeline() -> Pipeline:
    """Return a sklearn Pipeline that imputes then one-hot encodes categoricals.

    Missing values are filled with the most frequent category before encoding.

    Returns
    -------
    Pipeline
        Two-step pipeline: most-frequent imputation followed by OHE.
    """
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )


def build_preprocessing_pipeline() -> ColumnTransformer:
    """Build a full ColumnTransformer preprocessing pipeline.

    Applies median imputation + standard scaling to numeric features and
    most-frequent imputation + one-hot encoding to categorical features.
    The constants ``NUMERIC_FEATURES`` and ``CATEGORICAL_FEATURES`` drive
    the column selection.

    Returns
    -------
    ColumnTransformer
        Unfitted transformer ready to be ``fit_transform``-ed on training data.
    """
    logger.debug(
        "Building preprocessing pipeline with %d numeric and %d categorical features",
        len(NUMERIC_FEATURES),
        len(CATEGORICAL_FEATURES),
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", build_numeric_pipeline(), NUMERIC_FEATURES),
            ("categorical", build_categorical_pipeline(), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split.

    Parameters
    ----------
    X:
        Feature matrix.
    y:
        Binary target series.
    test_size:
        Fraction of data reserved for the test set.
    random_state:
        Seed for reproducibility.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        X_train, X_test, y_train, y_test
    """
    return train_test_split(  # type: ignore[return-value]
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
