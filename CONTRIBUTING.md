# Contributing to MedPredict

Thank you for taking the time to contribute. This guide covers how to set up a
local development environment, the coding standards enforced by CI, and the
commit and pull-request conventions used in this project.

---

## Development setup

### Prerequisites

| Tool | Minimum version |
|------|----------------|
| Python | 3.11 |
| Docker + Docker Compose | 24 |
| PostgreSQL (optional, Docker recommended) | 15 |

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/your-org/medpredict.git
cd medpredict
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
```

### 2. Install in editable mode with dev extras

```bash
pip install -e ".[dev]"
```

This installs all runtime and development dependencies declared in
`pyproject.toml` (pytest, ruff, mypy, httpx, etc.) and registers the
`medpredict-train` and `medpredict-serve` console scripts.

### 3. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and set at minimum:

```
DATABASE_URL=postgresql://postgres:<password>@localhost:5432/medpredict
MLFLOW_TRACKING_URI=http://localhost:5000
MODEL_PATH=models/best_model.pkl
POSTGRES_PASSWORD=<password>
```

### 4. Start backing services (optional)

If you want to run the full stack locally without Docker Compose:

```bash
# PostgreSQL
docker run -d --name medpredict-pg \
  -e POSTGRES_DB=medpredict \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=secret \
  -p 5432:5432 postgres:15

# MLflow
mlflow server --host 127.0.0.1 --port 5000
```

Or start everything at once:

```bash
docker compose up --build
```

---

## Common development commands

```bash
# Run all tests with coverage
pytest

# Run a specific module's tests
pytest tests/data/

# Lint (auto-fix safe issues)
ruff check --fix .

# Format
ruff format .

# Type check (strict mode)
mypy src/

# Train all baseline models
python -m src.training.train

# Serve the API locally
uvicorn src.api.main:app --reload --port 8000
```

---

## Coding standards

All of the following are enforced by CI. Pull requests that fail any check will
not be merged.

### Type annotations

Every function and method must have complete type annotations. Run
`mypy src/` with zero errors before opening a PR.

```python
# correct
def predict(features: pd.DataFrame) -> dict[str, float]:
    ...

# wrong — missing return type
def predict(features):
    ...
```

### Docstrings

Every public function, method, and class must have a docstring. Use
NumPy-style docstrings (Parameters / Returns / Raises sections), which is
the convention already established in the codebase.

```python
def split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> tuple[...]:
    """Stratified train/test split.

    Parameters
    ----------
    X:
        Feature matrix.
    y:
        Binary target series.
    test_size:
        Fraction of data reserved for the test set.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        X_train, X_test, y_train, y_test
    """
```

### Imports

- Standard library first, then third-party, then local (`src.*`).
- Use absolute imports: `from src.data.preprocess import binarise_target`.
- `ruff` enforces import order automatically; run `ruff check --fix .`.

### Formatting

Line length is 88 characters (ruff default). Run `ruff format .` before
committing; do not manually re-wrap lines.

### Pydantic

Use Pydantic v2 with `model_config = ConfigDict(...)`. Do not use v1-style
`class Config`.

### SQLAlchemy

Use the SQLAlchemy 2.0 API (`select()`, `AsyncSession`, `mapped_column`).
Never use the legacy `session.query()` API.

### Logging

Use the stdlib `logging` module. Do not use `print()` for diagnostic output.

```python
import logging
logger = logging.getLogger(__name__)
logger.info("Training %s ...", model_type)
```

### MLflow

Log parameters, metrics, and artefacts inside a `with mlflow.start_run():`
block. Tag every run:

```python
mlflow.set_tags({"model_type": ..., "dataset_version": ...})
```

### Environment variables

Never hard-code secrets or environment-specific paths. Read them with
`os.getenv()` (backed by `python-dotenv`) and document new variables in
`.env.example` and the `README.md` environment table.

---

## Testing

- One test file per source module: `tests/data/test_preprocess.py` mirrors
  `src/data/preprocess.py`.
- Mock external I/O (database, HTTP, filesystem). Do not hit real services in
  unit tests.
- Minimum coverage enforced by CI: **80 %**.
- Use `httpx.AsyncClient` (via `pytest-asyncio`) for API integration tests.

---

## Commit conventions

This project follows [Conventional Commits](https://www.conventionalcommits.org/).

```
<type>(<scope>): <short summary>

[optional body]

[optional footer]
```

### Types

| Type | When to use |
|------|-------------|
| `feat` | New feature or capability |
| `fix` | Bug fix |
| `refactor` | Code change that is neither a fix nor a feature |
| `test` | Adding or correcting tests |
| `docs` | Documentation only |
| `chore` | Tooling, dependency updates, config |
| `ci` | CI/CD pipeline changes |
| `perf` | Performance improvement |

### Examples

```
feat(api): add /predict endpoint with Pydantic v2 schema
fix(training): correct label encoding for multi-class targets
docs(readme): add dataset feature table and citation
chore(deps): bump fastapi to 0.111
test(data): add unit tests for engineer_features
```

### Rules

- Use the imperative mood in the summary: "add", not "added" or "adds".
- Keep the summary line under 72 characters.
- Reference issues or PRs in the footer: `Closes #42`.

---

## Branch naming

```
feat/<short-description>
fix/<short-description>
chore/<short-description>
docs/<short-description>
```

Examples: `feat/calibrated-probabilities`, `fix/missing-value-imputation`,
`docs/model-card`.

---

## Pull request checklist

Before opening a PR, verify all items below:

- [ ] `ruff check .` passes with zero errors
- [ ] `ruff format --check .` passes
- [ ] `mypy src/` passes with zero errors
- [ ] `pytest` passes with coverage >= 80 %
- [ ] New public functions and classes have docstrings
- [ ] `.env.example` updated if new environment variables were added
- [ ] `README.md` updated if the public interface changed
- [ ] No secrets or real patient data committed
