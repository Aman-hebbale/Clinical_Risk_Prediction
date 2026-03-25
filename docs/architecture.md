# Architecture

## Component diagram

```
┌─────────────────────────────────────────────────────────┐
│                        Docker Compose                    │
│                                                          │
│  ┌──────────────┐    ┌──────────────┐   ┌───────────┐   │
│  │   FastAPI     │    │  PostgreSQL  │   │  MLflow   │   │
│  │   (port 8000) │───►│  (port 5432) │   │ (port 5000│   │
│  └──────┬───────┘    └──────────────┘   └───────────┘   │
│         │                                      ▲          │
│         │  load model                          │          │
│         ▼                                      │          │
│  ┌──────────────┐                              │          │
│  │  models/     │      training run ───────────┘          │
│  │  best_model  │                                         │
│  └──────────────┘                                         │
└─────────────────────────────────────────────────────────┘
```

## Data flow

1. `src/training/train.py` fetches the UCI Heart Disease dataset via `ucimlrepo`.
2. Features are cleaned and split (`src/data/preprocess.py`).
3. A `GradientBoostingClassifier` pipeline is trained and evaluated.
4. The model is serialised to `models/best_model.pkl` and logged to MLflow.
5. At runtime, `src/api/main.py` loads the model once and serves predictions
   via `POST /predict`.
6. Each prediction can optionally be logged to the PostgreSQL database for
   audit and retraining purposes.

## Key design decisions

| Decision | Rationale |
|----------|-----------|
| Pydantic v2 frozen models | Immutable request/response objects prevent accidental mutation |
| `@lru_cache` on model load | Avoid reloading from disk on every request |
| SQLAlchemy 2.0 async | Non-blocking DB writes do not slow down the prediction path |
| MLflow model registry | Enables model versioning and stage promotion (staging → production) |
