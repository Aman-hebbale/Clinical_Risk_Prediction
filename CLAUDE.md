# CLAUDE.md — Development guidelines for Claude Code

This file tells Claude Code how to work in the MedPredict repository.
Read it before making any changes.

---

## Project summary

MedPredict is a Python 3.11 machine-learning service that predicts heart
disease risk from clinical features. The stack is:

- **Data**: UCI Heart Disease dataset fetched via `ucimlrepo`
- **ML**: scikit-learn pipelines + optional PyTorch model
- **API**: FastAPI + Pydantic v2, served with Uvicorn
- **Tracking**: MLflow (experiments + model registry)
- **Storage**: PostgreSQL via SQLAlchemy 2.0 (async)
- **Tooling**: ruff (lint/format), mypy strict, pytest + pytest-cov

---

## Repository layout

```
src/
  data/       – download.py, preprocess.py, features.py
  models/     – base.py, sklearn_model.py, torch_model.py, registry.py
  api/        – main.py, routers/, schemas.py, dependencies.py
  training/   – train.py, evaluate.py, experiment.py
tests/        – mirrors src/ layout; one test file per source module
notebooks/    – EDA only; no production logic
docs/         – architecture diagrams, ADRs
models/       – serialised artefacts (git-ignored)
```

---

## Coding conventions

### Type hints

Every function and method **must** have full type annotations.
Run `mypy src/` and resolve all errors before committing.

```python
# correct
def predict(features: pd.DataFrame) -> dict[str, float]:
    ...

# wrong — missing return type
def predict(features):
    ...
```

### Imports

- Standard library first, then third-party, then local (`src.*`).
- Use absolute imports from the package root (`from src.data.preprocess import ...`).
- `ruff` enforces import order (isort rules); run `ruff check --fix .` to auto-sort.

### Formatting

- Line length: **88** characters (ruff default).
- Run `ruff format .` before committing — do not manually re-wrap lines.

### Naming

| Kind | Convention | Example |
|------|-----------|---------|
| Module | `snake_case` | `feature_engineering.py` |
| Class | `PascalCase` | `HeartDiseaseModel` |
| Function / variable | `snake_case` | `load_dataset()` |
| Constant | `UPPER_SNAKE` | `DEFAULT_THRESHOLD` |
| Private | leading `_` | `_normalise_columns()` |

### Pydantic models

Use Pydantic v2 (`model_config = ConfigDict(...)`) — not v1-style `class Config`.

```python
from pydantic import BaseModel, ConfigDict

class PredictRequest(BaseModel):
    model_config = ConfigDict(frozen=True)
    age: int
    ...
```

### SQLAlchemy

Use the **2.0 style** (`select()`, `AsyncSession`, `mapped_column`).
Never use the legacy `session.query()` API.

### MLflow

Log parameters, metrics, and artefacts inside a `with mlflow.start_run():` block.
Tag every run with `mlflow.set_tags({"model_type": ..., "dataset_version": ...})`.

---

## Environment variables

All secrets and environment-specific values come from `.env` (git-ignored).
Use `python-dotenv` to load them; never hard-code values.

| Variable | Purpose |
|----------|---------|
| `DATABASE_URL` | PostgreSQL connection string |
| `MLFLOW_TRACKING_URI` | MLflow server address |
| `MODEL_PATH` | Path to the serialised model file |

---

## Testing

- One test file per source module: `tests/data/test_preprocess.py` mirrors
  `src/data/preprocess.py`.
- Use `pytest` fixtures; avoid global state.
- Mock external I/O (database, HTTP, filesystem) — do not hit real services
  in unit tests.
- Minimum coverage: **80 %** (enforced by `pytest --cov`).
- Use `httpx.AsyncClient` (via `pytest-asyncio`) for API integration tests.

```bash
pytest                          # run all tests with coverage
pytest tests/data/              # run a subset
pytest -k "test_preprocess"     # filter by name
```

---

## Git workflow

### Commit messages — Conventional Commits

```
<type>(<scope>): <short summary>

[optional body]

[optional footer]
```

Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `ci`, `perf`

Examples:

```
feat(api): add /predict endpoint with Pydantic v2 schema
fix(training): correct label encoding for multi-class targets
chore(deps): bump fastapi to 0.111
```

### Branch naming

```
feat/<short-description>
fix/<short-description>
chore/<short-description>
```

### PR checklist

Before opening a pull request verify:

- [ ] `ruff check .` passes with zero errors
- [ ] `ruff format --check .` passes
- [ ] `mypy src/` passes with zero errors
- [ ] `pytest` passes with coverage >= 80 %
- [ ] New public functions have docstrings
- [ ] `.env.example` updated if new env vars were added

---

## Common commands

```bash
# Install in editable mode with dev extras
pip install -e ".[dev]"

# Lint (auto-fix safe issues)
ruff check --fix .

# Format
ruff format .

# Type check
mypy src/

# Tests with coverage
pytest

# Train model
python -m src.training.train

# Serve API locally
uvicorn src.api.main:app --reload --port 8000

# Start all services (API + DB + MLflow)
docker compose up --build
```

---

## What Claude should NOT do

- Do not commit secrets or real patient data.
- Do not bypass `mypy` errors with `# type: ignore` without a comment explaining why.
- Do not add new dependencies without updating `pyproject.toml` and
  discussing the addition in the PR description.
- Do not use `print()` for logging — use the stdlib `logging` module.
- Do not write synchronous blocking I/O in async FastAPI route handlers.
- Do not hard-code file paths; derive them from `MODEL_PATH` or `pathlib.Path`.
