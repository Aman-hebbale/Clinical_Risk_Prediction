"""Unit tests for src.data.preprocess."""

import numpy as np
import pandas as pd
import pytest

from src.data.preprocess import binarise_target, drop_missing, split


@pytest.fixture()
def sample_X() -> pd.DataFrame:
    """Minimal feature matrix with no missing values."""
    return pd.DataFrame(
        {
            "age": [45, 55, 60, 35, 50],
            "chol": [200, 220, 180, 240, 260],
        }
    )


@pytest.fixture()
def sample_y() -> pd.Series:
    """Target series with multi-class labels."""
    return pd.Series([0, 1, 2, 0, 3])


def test_binarise_target_zero_stays_zero(sample_y: pd.Series) -> None:
    result = binarise_target(sample_y)
    assert result[0] == 0


def test_binarise_target_nonzero_becomes_one(sample_y: pd.Series) -> None:
    result = binarise_target(sample_y)
    assert result[1] == 1
    assert result[2] == 1
    assert result[4] == 1


def test_drop_missing_removes_nan_rows() -> None:
    X = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [4.0, 5.0, 6.0]})
    y = pd.Series([0, 1, 0])
    X_clean, y_clean = drop_missing(X, y)
    assert len(X_clean) == 2
    assert len(y_clean) == 2


def test_split_sizes(sample_X: pd.DataFrame, sample_y: pd.Series) -> None:
    y_binary = binarise_target(sample_y)
    X_train, X_test, y_train, y_test = split(sample_X, y_binary, test_size=0.4)
    assert len(X_train) + len(X_test) == len(sample_X)
