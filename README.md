# MedPredict

![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)
![License MIT](https://img.shields.io/badge/license-MIT-green.svg)
![CI](https://img.shields.io/github/actions/workflow/status/your-org/medpredict/ci.yml?label=CI)

MedPredict — Heart disease prediction API using UCI Heart Disease dataset,
served via FastAPI, tracked with MLflow, containerized with Docker.

---

## Overview

MedPredict trains a classification model on the
[UCI Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)
and exposes predictions through a production-ready REST API. All experiments
are tracked in MLflow; the service is packaged as a Docker container.

### Key capabilities

- Automated data download via `ucimlrepo`
- Feature preprocessing pipeline with scikit-learn
- PyTorch neural network alongside sklearn baselines
- FastAPI prediction endpoint with Pydantic v2 schemas
- MLflow experiment tracking and model registry
- PostgreSQL persistence for prediction logs
- Full type-annotated codebase (mypy strict)
- Ruff linting + pytest + coverage enforced in CI

---

## Architecture

```
medpredict/
├── src/
│   ├── data/        # Dataset download, preprocessing, feature engineering
│   ├── models/      # Model definitions and serialization helpers
│   ├── api/         # FastAPI app, routers, Pydantic schemas
│   └── training/    # Training pipelines, evaluation, MLflow logging
├── tests/           # pytest test suite
├── notebooks/       # Exploratory analysis (Jupyter)
├── docs/            # Architecture diagrams, API reference
├── models/          # Serialized model artefacts (git-ignored)
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

Request flow:

```
Client
  └─► POST /predict  (FastAPI)
          └─► load model from MODEL_PATH
                  └─► sklearn pipeline  ──►  response JSON
                          └─► log to PostgreSQL
                                  └─► MLflow tracking (optional)
```

---

## Prerequisites

| Tool | Minimum version |
|------|----------------|
| Python | 3.11 |
| PostgreSQL | 15 |
| Docker + Compose | 24 |

---

## Quick start

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/your-org/medpredict.git
cd medpredict
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your DATABASE_URL, MLFLOW_TRACKING_URI, MODEL_PATH
```

### 3. Train the model

```bash
medpredict-train
# or: python -m src.training.train
```

### 4. Serve the API

```bash
medpredict-serve
# or: uvicorn src.api.main:app --reload
```

API docs available at `http://localhost:8000/docs`.

### 5. Run with Docker

```bash
docker compose up --build
```

---

## Development

### Run tests

```bash
pytest
```

### Lint and format

```bash
ruff check .
ruff format .
```

### Type check

```bash
mypy src/
```

---

## API reference

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness probe |
| `POST` | `/predict` | Return heart disease probability |
| `GET` | `/model/info` | Active model metadata |

### Example request

```json
POST /predict
{
  "age": 54,
  "sex": 1,
  "cp": 0,
  "trestbps": 130,
  "chol": 245,
  "fbs": 0,
  "restecg": 0,
  "thalach": 155,
  "exang": 0,
  "oldpeak": 1.4,
  "slope": 2,
  "ca": 0,
  "thal": 2
}
```

### Example response

```json
{
  "prediction": 0,
  "probability": 0.12,
  "model_version": "1.0.0"
}
```

---

## License

MIT — see [LICENSE](LICENSE).
