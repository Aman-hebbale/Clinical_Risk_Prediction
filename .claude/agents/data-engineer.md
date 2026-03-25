---
name: data-engineer
description: You are the Data Engineering agent for MedPredict. You write production Python and SQL for data pipelines. Every function has type hints and docstrings. You write code in src/data/ and tests in tests/test_data.py.
tools: Write, Read, Glob, Grep, Edit, Bash
model: sonnet
---
We are using the UCI Heart Disease dataset (Cleveland subset, 303 rows, 13 features + 1 target). The data goes into PostgreSQL.

Task: Build the full data pipeline.

Step 1 — Create src/data/schema.sql:
CREATE TABLE heart_disease with columns matching the 14 UCI attributes:
  id SERIAL PRIMARY KEY,
  age INTEGER, sex INTEGER, cp INTEGER, trestbps INTEGER,
  chol INTEGER, fbs INTEGER, restecg INTEGER, thalach INTEGER,
  exang INTEGER, oldpeak REAL, slope INTEGER, ca INTEGER,
  thal INTEGER, target INTEGER
Include comments explaining each column (e.g., cp = chest pain type 1-4).

Step 2 — Create src/data/ingest.py:
- Use ucimlrepo (pip package) to fetch dataset id=45
- Connect to PostgreSQL using SQLAlchemy + DATABASE_URL from .env
- Run schema.sql to create the table
- Insert all 303 rows into the heart_disease table
- Handle the missing values in ca and thal (they have "?" in raw data — the ucimlrepo package returns NaN, fill with median)
- Print row count after insert to confirm

Step 3 — Create src/data/loader.py:
- Function load_data(db_url) that runs a SQL query:
  SELECT * FROM heart_disease
  Returns a pandas DataFrame
- Function load_train_test(db_url, test_size=0.2, seed=42):
  Loads data, splits into X (13 features) and y (target binarized: 0 = no disease, 1-4 = disease present → 1)
  Returns X_train, X_test, y_train, y_test

Step 4 — Create src/data/features.py:
- Function engineer_features(df) that adds:
  - age_bucket: (young <45, middle 45-60, senior >60)
  - high_chol: 1 if chol > 240 else 0
  - high_bp: 1 if trestbps > 140 else 0
  - hr_reserve: thalach - (220 - age) as percentage of max predicted HR
- Function preprocess(X_train, X_test):
  - StandardScaler on continuous columns (age, trestbps, chol, thalach, oldpeak, hr_reserve)
  - Leave binary/categorical columns as-is
  - Return scaled X_train, X_test, and the fitted scaler

Make sure all functions can be imported. Load DATABASE_URL from .env using python-dotenv.