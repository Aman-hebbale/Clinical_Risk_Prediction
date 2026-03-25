"""Feature engineering helpers."""

import pandas as pd


def add_age_group(X: pd.DataFrame) -> pd.DataFrame:
    """Add an ordinal *age_group* column binned into decades.

    Parameters
    ----------
    X:
        Input feature matrix.  Must contain an *age* column.

    Returns
    -------
    pd.DataFrame
        Copy of *X* with the additional *age_group* column.
    """
    out: pd.DataFrame = X.copy()
    out["age_group"] = pd.cut(
        out["age"],
        bins=[0, 40, 50, 60, 70, 200],
        labels=[0, 1, 2, 3, 4],
        right=False,
    ).astype(int)
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
