"""Feature engineering helpers."""

import logging

import pandas as pd

logger: logging.Logger = logging.getLogger(__name__)


def add_age_group(X: pd.DataFrame) -> pd.DataFrame:
    """Add an ordinal *age_group* column binned into decades.

    Parameters
    ----------
    X:
        Input feature matrix.  Must contain an *age* column.

    Returns
    -------
    pd.DataFrame
        Copy of *X* with the additional *age_group* column (values 0-4).
    """
    out: pd.DataFrame = X.copy()
    out["age_group"] = pd.cut(
        out["age"],
        bins=[0, 40, 50, 60, 70, 200],
        labels=[0, 1, 2, 3, 4],
        right=False,
    ).astype(int)
    return out


def compute_chol_age_ratio(X: pd.DataFrame) -> pd.DataFrame:
    """Add a *chol_age_ratio* column equal to ``chol / age``.

    A higher ratio indicates elevated cholesterol relative to the patient's
    age, which is a clinically relevant risk signal.

    Parameters
    ----------
    X:
        Input feature matrix.  Must contain *chol* and *age* columns.

    Returns
    -------
    pd.DataFrame
        Copy of *X* with the additional *chol_age_ratio* column.

    Notes
    -----
    Rows where *age* is zero would produce ``inf``; the UCI dataset contains
    no such rows, but callers should ensure *age* > 0 before calling this
    function.
    """
    out: pd.DataFrame = X.copy()
    out["chol_age_ratio"] = out["chol"] / out["age"]
    logger.debug("Computed chol_age_ratio; NaN count: %d", out["chol_age_ratio"].isna().sum())
    return out


def engineer_features(X: pd.DataFrame) -> pd.DataFrame:
    """Orchestrate all feature engineering steps in a single call.

    Applies the following transformations in order:

    1. :func:`add_age_group` — ordinal age bucket column.
    2. :func:`compute_chol_age_ratio` — cholesterol-to-age ratio column.

    Parameters
    ----------
    X:
        Raw feature matrix.  Must contain *age* and *chol* columns.

    Returns
    -------
    pd.DataFrame
        Feature matrix augmented with *age_group* and *chol_age_ratio*.
    """
    logger.info("Running feature engineering on %d rows", len(X))
    out: pd.DataFrame = add_age_group(X)
    out = compute_chol_age_ratio(out)
    return out


def select_features(X: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Return a dataframe containing only the requested columns.

    Parameters
    ----------
    X:
        Full feature matrix.
    feature_names:
        Columns to retain.

    Returns
    -------
    pd.DataFrame
        Subset of *X*.
    """
    return X[feature_names].copy()
