"""Download the UCI Heart Disease dataset using ucimlrepo."""

import logging
from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo  # type: ignore[import-untyped]

logger: logging.Logger = logging.getLogger(__name__)

UCI_HEART_DISEASE_ID: int = 45


def fetch_dataset(cache_dir: Path | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Fetch the UCI Heart Disease dataset.

    Parameters
    ----------
    cache_dir:
        Optional directory to cache the raw CSV files.  If *None* the data
        is returned without writing to disk.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        Feature matrix X and target series y.
    """
    logger.info("Fetching UCI Heart Disease dataset (id=%d)", UCI_HEART_DISEASE_ID)
    heart_disease = fetch_ucirepo(id=UCI_HEART_DISEASE_ID)

    X: pd.DataFrame = heart_disease.data.features
    y: pd.Series = heart_disease.data.targets.squeeze()

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        X.to_csv(cache_dir / "features.csv", index=False)
        y.to_csv(cache_dir / "targets.csv", index=False)
        logger.info("Cached dataset to %s", cache_dir)

    return X, y
