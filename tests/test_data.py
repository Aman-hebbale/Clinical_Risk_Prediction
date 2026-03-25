"""Comprehensive tests for the MedPredict data pipeline.

Covers:
- src.data.preprocess.build_preprocessing_pipeline
- src.data.features.compute_chol_age_ratio
- src.data.features.engineer_features
- src.data.pipeline.run_pipeline  (fetch_dataset mocked)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from src.data.features import compute_chol_age_ratio, engineer_features
from src.data.preprocess import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_preprocessing_pipeline,
)
from src.data.pipeline import PipelineResult, run_pipeline


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def minimal_X() -> pd.DataFrame:
    """Feature matrix with the 13 UCI columns + engineered columns absent.

    Contains two rows with missing values in *ca* and *thal* to exercise
    imputation inside the preprocessing pipeline.
    """
    return pd.DataFrame(
        {
            "age": [45, 55, 60, 35, 50, 62, 48],
            "sex": [1, 0, 1, 1, 0, 1, 0],
            "cp": [3, 2, 1, 4, 2, 3, 1],
            "trestbps": [130, 140, 120, 110, 150, 135, 125],
            "chol": [250, 200, 180, 240, 260, 220, 190],
            "fbs": [0, 1, 0, 0, 1, 0, 1],
            "restecg": [1, 0, 2, 0, 1, 0, 1],
            "thalach": [150, 160, 140, 170, 130, 155, 145],
            "exang": [0, 1, 0, 0, 1, 0, 1],
            "oldpeak": [1.5, 2.0, 0.5, 0.0, 3.0, 1.0, 2.5],
            "slope": [1, 2, 1, 3, 2, 1, 2],
            "ca": [0.0, np.nan, 1.0, 0.0, 2.0, 1.0, np.nan],
            "thal": [3.0, 6.0, np.nan, 7.0, 3.0, np.nan, 6.0],
        }
    )


@pytest.fixture()
def clean_X() -> pd.DataFrame:
    """Feature matrix with no missing values — for feature engineering tests."""
    return pd.DataFrame(
        {
            "age": [45, 55, 60, 35, 50],
            "sex": [1, 0, 1, 1, 0],
            "cp": [3, 2, 1, 4, 2],
            "trestbps": [130, 140, 120, 110, 150],
            "chol": [250, 200, 180, 240, 260],
            "fbs": [0, 1, 0, 0, 1],
            "restecg": [1, 0, 2, 0, 1],
            "thalach": [150, 160, 140, 170, 130],
            "exang": [0, 1, 0, 0, 1],
            "oldpeak": [1.5, 2.0, 0.5, 0.0, 3.0],
            "slope": [1, 2, 1, 3, 2],
            "ca": [0.0, 1.0, 1.0, 0.0, 2.0],
            "thal": [3.0, 6.0, 7.0, 7.0, 3.0],
        }
    )


@pytest.fixture()
def binary_y() -> pd.Series:
    """Binary target series aligned with *clean_X*."""
    return pd.Series([0, 1, 1, 0, 1], dtype=np.int8)


# ---------------------------------------------------------------------------
# build_preprocessing_pipeline
# ---------------------------------------------------------------------------


class TestBuildPreprocessingPipeline:
    """Tests for the ColumnTransformer returned by build_preprocessing_pipeline."""

    def test_returns_column_transformer(self) -> None:
        """Pipeline object has the expected sklearn type."""
        from sklearn.compose import ColumnTransformer

        ct = build_preprocessing_pipeline()
        assert isinstance(ct, ColumnTransformer)

    def test_output_has_no_nan(self, minimal_X: pd.DataFrame) -> None:
        """Imputation ensures no NaN values survive in the transformed output."""
        ct = build_preprocessing_pipeline()
        result: np.ndarray = ct.fit_transform(minimal_X)
        assert not np.isnan(result).any(), "NaN values found after preprocessing"

    def test_output_row_count_matches_input(self, minimal_X: pd.DataFrame) -> None:
        """Number of output rows equals number of input rows."""
        ct = build_preprocessing_pipeline()
        result: np.ndarray = ct.fit_transform(minimal_X)
        assert result.shape[0] == len(minimal_X)

    def test_numeric_features_are_scaled(self, minimal_X: pd.DataFrame) -> None:
        """The first *n* columns (numeric) have approximately zero mean after scaling."""
        ct = build_preprocessing_pipeline()
        result: np.ndarray = ct.fit_transform(minimal_X)
        n_numeric: int = len(NUMERIC_FEATURES)
        numeric_block: np.ndarray = result[:, :n_numeric]
        # StandardScaler guarantees mean ≈ 0 on the training data
        np.testing.assert_allclose(
            numeric_block.mean(axis=0),
            np.zeros(n_numeric),
            atol=1e-10,
        )

    def test_output_column_count_is_at_least_feature_count(
        self, minimal_X: pd.DataFrame
    ) -> None:
        """OHE expands categoricals, so output cols >= number of raw features."""
        ct = build_preprocessing_pipeline()
        result: np.ndarray = ct.fit_transform(minimal_X)
        assert result.shape[1] >= len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)

    def test_transform_matches_fit_transform_on_same_data(
        self, minimal_X: pd.DataFrame
    ) -> None:
        """Calling transform after fit produces the same result as fit_transform."""
        ct = build_preprocessing_pipeline()
        fit_transformed: np.ndarray = ct.fit_transform(minimal_X)
        transformed: np.ndarray = ct.transform(minimal_X)
        np.testing.assert_array_equal(fit_transformed, transformed)


# ---------------------------------------------------------------------------
# compute_chol_age_ratio
# ---------------------------------------------------------------------------


class TestComputeCholAgeRatio:
    """Tests for compute_chol_age_ratio."""

    def test_column_added(self, clean_X: pd.DataFrame) -> None:
        """Output dataframe contains the new column."""
        result = compute_chol_age_ratio(clean_X)
        assert "chol_age_ratio" in result.columns

    def test_formula_is_correct(self, clean_X: pd.DataFrame) -> None:
        """chol_age_ratio equals chol divided by age for every row."""
        result = compute_chol_age_ratio(clean_X)
        expected: pd.Series = clean_X["chol"] / clean_X["age"]
        pd.testing.assert_series_equal(
            result["chol_age_ratio"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_no_nan_in_output(self, clean_X: pd.DataFrame) -> None:
        """No NaN values in the ratio column when inputs are clean."""
        result = compute_chol_age_ratio(clean_X)
        assert not result["chol_age_ratio"].isna().any()

    def test_original_dataframe_not_mutated(self, clean_X: pd.DataFrame) -> None:
        """Source dataframe is not modified in place."""
        original_cols = list(clean_X.columns)
        compute_chol_age_ratio(clean_X)
        assert list(clean_X.columns) == original_cols

    def test_values_are_positive(self, clean_X: pd.DataFrame) -> None:
        """All ratio values are positive given positive chol and age."""
        result = compute_chol_age_ratio(clean_X)
        assert (result["chol_age_ratio"] > 0).all()


# ---------------------------------------------------------------------------
# engineer_features
# ---------------------------------------------------------------------------


class TestEngineerFeatures:
    """Tests for engineer_features."""

    def test_adds_age_group_column(self, clean_X: pd.DataFrame) -> None:
        """age_group column is present in the output."""
        result = engineer_features(clean_X)
        assert "age_group" in result.columns

    def test_adds_chol_age_ratio_column(self, clean_X: pd.DataFrame) -> None:
        """chol_age_ratio column is present in the output."""
        result = engineer_features(clean_X)
        assert "chol_age_ratio" in result.columns

    def test_both_columns_added_together(self, clean_X: pd.DataFrame) -> None:
        """Both engineered columns are present after a single call."""
        result = engineer_features(clean_X)
        assert {"age_group", "chol_age_ratio"}.issubset(result.columns)

    def test_row_count_unchanged(self, clean_X: pd.DataFrame) -> None:
        """engineer_features does not drop or duplicate rows."""
        result = engineer_features(clean_X)
        assert len(result) == len(clean_X)

    def test_original_columns_preserved(self, clean_X: pd.DataFrame) -> None:
        """All original columns are retained in the output."""
        original_cols = set(clean_X.columns)
        result = engineer_features(clean_X)
        assert original_cols.issubset(result.columns)

    def test_age_group_values_are_in_valid_range(self, clean_X: pd.DataFrame) -> None:
        """age_group ordinal labels are within [0, 4]."""
        result = engineer_features(clean_X)
        assert result["age_group"].between(0, 4).all()

    def test_chol_age_ratio_matches_direct_computation(
        self, clean_X: pd.DataFrame
    ) -> None:
        """chol_age_ratio from engineer_features equals chol / age directly."""
        result = engineer_features(clean_X)
        expected: pd.Series = clean_X["chol"] / clean_X["age"]
        pd.testing.assert_series_equal(
            result["chol_age_ratio"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )


# ---------------------------------------------------------------------------
# run_pipeline
# ---------------------------------------------------------------------------


def _make_mock_dataset(n_rows: int = 100) -> tuple[pd.DataFrame, pd.Series]:
    """Return a synthetic (X, y) pair that mimics the UCI Heart Disease schema."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "age": rng.integers(30, 75, size=n_rows).astype(float),
            "sex": rng.integers(0, 2, size=n_rows).astype(float),
            "cp": rng.integers(1, 5, size=n_rows).astype(float),
            "trestbps": rng.integers(90, 180, size=n_rows).astype(float),
            "chol": rng.integers(150, 350, size=n_rows).astype(float),
            "fbs": rng.integers(0, 2, size=n_rows).astype(float),
            "restecg": rng.integers(0, 3, size=n_rows).astype(float),
            "thalach": rng.integers(90, 200, size=n_rows).astype(float),
            "exang": rng.integers(0, 2, size=n_rows).astype(float),
            "oldpeak": rng.uniform(0, 5, size=n_rows),
            "slope": rng.integers(1, 4, size=n_rows).astype(float),
            "ca": rng.integers(0, 4, size=n_rows).astype(float),
            "thal": rng.choice([3.0, 6.0, 7.0], size=n_rows),
        }
    )
    y = pd.Series(rng.integers(0, 5, size=n_rows), name="num")
    return X, y


class TestRunPipeline:
    """Tests for run_pipeline with fetch_dataset mocked out."""

    @pytest.fixture()
    def mock_fetch(self) -> tuple[pd.DataFrame, pd.Series]:
        """Return the synthetic dataset used in all run_pipeline tests."""
        return _make_mock_dataset(n_rows=100)

    def test_returns_pipeline_result(
        self,
        mock_fetch: tuple[pd.DataFrame, pd.Series],
        tmp_path: Path,
    ) -> None:
        """run_pipeline returns a PipelineResult instance."""
        with patch("src.data.pipeline.fetch_dataset", return_value=mock_fetch):
            result = run_pipeline(
                raw_dir=tmp_path / "raw",
                processed_dir=tmp_path / "processed",
            )
        assert isinstance(result, PipelineResult)

    def test_npy_files_are_written(
        self,
        mock_fetch: tuple[pd.DataFrame, pd.Series],
        tmp_path: Path,
    ) -> None:
        """All four .npy artefacts exist after the pipeline runs."""
        processed_dir = tmp_path / "processed"
        with patch("src.data.pipeline.fetch_dataset", return_value=mock_fetch):
            run_pipeline(raw_dir=tmp_path / "raw", processed_dir=processed_dir)

        expected_files = [
            "X_train_processed.npy",
            "X_test_processed.npy",
            "y_train.npy",
            "y_test.npy",
        ]
        for fname in expected_files:
            assert (processed_dir / fname).exists(), f"{fname} not found"

    def test_train_test_shapes_align(
        self,
        mock_fetch: tuple[pd.DataFrame, pd.Series],
        tmp_path: Path,
    ) -> None:
        """X_train and y_train have the same number of rows; likewise for test."""
        with patch("src.data.pipeline.fetch_dataset", return_value=mock_fetch):
            result = run_pipeline(
                raw_dir=tmp_path / "raw",
                processed_dir=tmp_path / "processed",
            )

        assert result.X_train.shape[0] == len(result.y_train)
        assert result.X_test.shape[0] == len(result.y_test)

    def test_train_plus_test_equals_total(
        self,
        mock_fetch: tuple[pd.DataFrame, pd.Series],
        tmp_path: Path,
    ) -> None:
        """Combined train and test row counts equal the cleaned dataset size."""
        with patch("src.data.pipeline.fetch_dataset", return_value=mock_fetch):
            result = run_pipeline(
                raw_dir=tmp_path / "raw",
                processed_dir=tmp_path / "processed",
            )

        total = result.X_train.shape[0] + result.X_test.shape[0]
        # 100 rows, no missing values in synthetic data -> all rows survive
        assert total == 100

    def test_no_nan_in_processed_arrays(
        self,
        mock_fetch: tuple[pd.DataFrame, pd.Series],
        tmp_path: Path,
    ) -> None:
        """Imputation ensures no NaN survives in the processed feature arrays."""
        with patch("src.data.pipeline.fetch_dataset", return_value=mock_fetch):
            result = run_pipeline(
                raw_dir=tmp_path / "raw",
                processed_dir=tmp_path / "processed",
            )

        assert not np.isnan(result.X_train).any()
        assert not np.isnan(result.X_test).any()

    def test_saved_npy_matches_returned_arrays(
        self,
        mock_fetch: tuple[pd.DataFrame, pd.Series],
        tmp_path: Path,
    ) -> None:
        """Arrays saved to disk match the arrays in the returned PipelineResult."""
        processed_dir = tmp_path / "processed"
        with patch("src.data.pipeline.fetch_dataset", return_value=mock_fetch):
            result = run_pipeline(
                raw_dir=tmp_path / "raw",
                processed_dir=processed_dir,
            )

        np.testing.assert_array_equal(
            np.load(processed_dir / "X_train_processed.npy"),
            result.X_train,
        )
        np.testing.assert_array_equal(
            np.load(processed_dir / "X_test_processed.npy"),
            result.X_test,
        )

    def test_target_is_binary(
        self,
        mock_fetch: tuple[pd.DataFrame, pd.Series],
        tmp_path: Path,
    ) -> None:
        """All values in y_train and y_test are 0 or 1 after binarisation."""
        with patch("src.data.pipeline.fetch_dataset", return_value=mock_fetch):
            result = run_pipeline(
                raw_dir=tmp_path / "raw",
                processed_dir=tmp_path / "processed",
            )

        assert set(result.y_train.unique()).issubset({0, 1})
        assert set(result.y_test.unique()).issubset({0, 1})

    def test_pipeline_column_count_consistent(
        self,
        mock_fetch: tuple[pd.DataFrame, pd.Series],
        tmp_path: Path,
    ) -> None:
        """X_train and X_test have the same number of columns."""
        with patch("src.data.pipeline.fetch_dataset", return_value=mock_fetch):
            result = run_pipeline(
                raw_dir=tmp_path / "raw",
                processed_dir=tmp_path / "processed",
            )

        assert result.X_train.shape[1] == result.X_test.shape[1]

    def test_fetch_dataset_called_with_raw_dir(
        self,
        tmp_path: Path,
    ) -> None:
        """fetch_dataset is called exactly once with the supplied raw_dir."""
        mock_data = _make_mock_dataset(n_rows=80)
        raw_dir = tmp_path / "raw"
        with patch(
            "src.data.pipeline.fetch_dataset", return_value=mock_data
        ) as mock_fn:
            run_pipeline(raw_dir=raw_dir, processed_dir=tmp_path / "processed")

        mock_fn.assert_called_once_with(cache_dir=raw_dir)
