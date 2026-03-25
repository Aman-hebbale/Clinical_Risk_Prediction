"""End-to-end data pipeline orchestrator for the MedPredict project."""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

from src.data.download import fetch_dataset
from src.data.features import engineer_features
from src.data.preprocess import (
    binarise_target,
    build_preprocessing_pipeline,
    drop_missing,
    split,
)

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Container for all artefacts produced by :func:`run_pipeline`.

    Attributes
    ----------
    X_train:
        Processed training feature matrix (numpy array).
    X_test:
        Processed test feature matrix (numpy array).
    y_train:
        Binary training labels.
    y_test:
        Binary test labels.
    pipeline:
        Fitted :class:`~sklearn.compose.ColumnTransformer` used to transform
        both splits.  Persist this object to avoid data leakage when scoring
        new samples.
    """

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: pd.Series
    y_test: pd.Series
    pipeline: ColumnTransformer


def run_pipeline(
    raw_dir: Path,
    processed_dir: Path,
) -> PipelineResult:
    """Execute the full data pipeline from raw download to processed arrays.

    Steps
    -----
    1. Fetch the UCI Heart Disease dataset via :func:`~src.data.download.fetch_dataset`,
       caching raw CSVs to *raw_dir*.
    2. Binarise the target with :func:`~src.data.preprocess.binarise_target`.
    3. Drop rows with missing values via :func:`~src.data.preprocess.drop_missing`.
    4. Engineer additional features with :func:`~src.data.features.engineer_features`.
    5. Perform a stratified train/test split via :func:`~src.data.preprocess.split`.
    6. Build :func:`~src.data.preprocess.build_preprocessing_pipeline`, fit on the
       training split, and transform both splits.
    7. Persist ``X_train_processed``, ``X_test_processed``, ``y_train``, and
       ``y_test`` as ``.npy`` files inside *processed_dir*.

    Parameters
    ----------
    raw_dir:
        Directory where the raw feature and target CSVs will be cached.
    processed_dir:
        Directory where the four processed ``.npy`` artefacts will be saved.

    Returns
    -------
    PipelineResult
        Dataclass holding the four processed arrays and the fitted pipeline.
    """
    logger.info("Starting data pipeline")

    # Step 1 — download / load from cache
    X_raw, y_raw = fetch_dataset(cache_dir=raw_dir)
    logger.info("Loaded dataset: X=%s, y=%s", X_raw.shape, y_raw.shape)

    # Step 2 — binarise target
    y_bin: pd.Series = binarise_target(y_raw)

    # Step 3 — drop missing
    X_clean, y_clean = drop_missing(X_raw, y_bin)
    logger.info("After drop_missing: %d rows remain", len(X_clean))

    # Step 4 — engineer features
    X_eng: pd.DataFrame = engineer_features(X_clean)
    logger.info("Feature matrix columns after engineering: %s", list(X_eng.columns))

    # Step 5 — train/test split
    X_train_df, X_test_df, y_train, y_test = split(X_eng, y_clean)
    logger.info(
        "Split sizes — train: %d, test: %d", len(X_train_df), len(X_test_df)
    )

    # Step 6 — fit preprocessing pipeline on train; transform both splits
    preprocessor: ColumnTransformer = build_preprocessing_pipeline()
    X_train_processed: np.ndarray = preprocessor.fit_transform(X_train_df)
    X_test_processed: np.ndarray = preprocessor.transform(X_test_df)
    logger.info(
        "Processed shapes — train: %s, test: %s",
        X_train_processed.shape,
        X_test_processed.shape,
    )

    # Step 7 — persist artefacts
    processed_dir.mkdir(parents=True, exist_ok=True)
    np.save(processed_dir / "X_train_processed.npy", X_train_processed)
    np.save(processed_dir / "X_test_processed.npy", X_test_processed)
    np.save(processed_dir / "y_train.npy", y_train.to_numpy())
    np.save(processed_dir / "y_test.npy", y_test.to_numpy())
    logger.info("Saved processed artefacts to %s", processed_dir)

    return PipelineResult(
        X_train=X_train_processed,
        X_test=X_test_processed,
        y_train=y_train,
        y_test=y_test,
        pipeline=preprocessor,
    )
