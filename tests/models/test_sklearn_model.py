"""Unit tests for SklearnModel."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.sklearn_model import SklearnModel


@pytest.fixture()
def trained_model() -> SklearnModel:
    """Return a model trained on a tiny synthetic dataset."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((60, 13)), columns=[f"f{i}" for i in range(13)])
    y = pd.Series((rng.random(60) > 0.5).astype(int))
    return SklearnModel(n_estimators=10).fit(X, y)


def test_predict_shape(trained_model: SklearnModel) -> None:
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.standard_normal((10, 13)), columns=[f"f{i}" for i in range(13)])
    preds = trained_model.predict(X)
    assert preds.shape == (10,)


def test_predict_proba_shape(trained_model: SklearnModel) -> None:
    rng = np.random.default_rng(2)
    X = pd.DataFrame(rng.standard_normal((10, 13)), columns=[f"f{i}" for i in range(13)])
    proba = trained_model.predict_proba(X)
    assert proba.shape == (10, 2)


def test_save_and_load(trained_model: SklearnModel) -> None:
    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.standard_normal((5, 13)), columns=[f"f{i}" for i in range(13)])
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.pkl"
        trained_model.save(path)
        loaded = SklearnModel.load(path)
    np.testing.assert_array_equal(trained_model.predict(X), loaded.predict(X))
