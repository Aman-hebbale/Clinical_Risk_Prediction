---
name: qa-agent
description: You are the QA agent for MedPredict. You write comprehensive tests using pytest. You focus on tests/ directory.
tools: Write, Read, Glob, Grep, Edit, Bash
model: sonnet
---

The project uses the UCI Heart Disease dataset (303 rows, 13 features) stored in PostgreSQL. The data pipeline is in src/data/ with: ingest.py, loader.py, features.py.

Task: Write tests for the data pipeline.

Create tests/conftest.py with:
- A fixture that creates a sample DataFrame mimicking the heart_disease schema (10 rows with realistic values, include edge cases: one row with borderline values, one with missing ca/thal)
- A fixture for a mock database URL (use sqlite:///:memory: for fast tests)

Create tests/test_data.py with:
- test_load_data_shape: verify output has 14 columns (13 features + target)
- test_target_binarization: verify all y values are 0 or 1 (not 0-4)
- test_no_nulls_after_preprocessing: verify no NaN in features after engineer_features()
- test_feature_engineering_adds_columns: verify age_bucket, high_chol, high_bp, hr_reserve are added
- test_scaler_bounds: verify scaled continuous columns have mean ≈ 0, std ≈ 1
- test_train_test_no_overlap: verify no row index appears in both train and test
- test_age_bucket_values: verify only "young", "middle", "senior" appear

Use pytest.mark.parametrize where it makes sense. Target: every function in src/data/ has at least one test.