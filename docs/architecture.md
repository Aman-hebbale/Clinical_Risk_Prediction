# Architecture

## Component overview

```mermaid
flowchart TB
    subgraph Data["Data layer  (src/data/)"]
        DL[download.py<br>fetch_dataset]
        PRE[preprocess.py<br>binarise_target<br>build_preprocessing_pipeline<br>split]
        FE[features.py<br>engineer_features]
        PL[pipeline.py<br>run_pipeline]
        DL --> PRE --> FE --> PL
    end

    subgraph Models["Models  (src/models/)"]
        BASE[base.py<br>BaseModel ABC]
        LR[logistic_model.py<br>LogisticModel]
        RF[random_forest_model.py<br>RandomForestModel]
        GB[sklearn_model.py<br>SklearnModel<br>GradientBoosting]
        NN[torch_model.py<br>TorchModel MLP]
        REG[registry.py<br>load_model]
        BASE --> LR & RF & GB & NN
        REG --> BASE
    end

    subgraph Training["Training  (src/training/)"]
        TRAIN[train.py<br>main]
        EVAL[evaluate.py<br>full_report]
    end

    subgraph API["API  (src/api/)"]
        MAIN[main.py<br>FastAPI app]
        SCH[schemas.py<br>PredictRequest<br>PredictResponse]
        DEP[dependencies.py<br>get_model]
        ROUTER[routers/predict.py<br>POST /predict]
        MAIN --> DEP --> REG
        MAIN --> ROUTER --> SCH
    end

    subgraph Infra["Infrastructure"]
        PG[(PostgreSQL)]
        MLF[MLflow server]
        DOCKER[Docker Compose]
    end

    PL --> TRAIN
    TRAIN --> Models
    TRAIN --> MLF
    TRAIN --> Disk[models/*.pkl]
    Disk --> REG
    API --> PG
    DOCKER --> API & PG & MLF
```

## Data flow: UCI download to trained model

```mermaid
sequenceDiagram
    participant S as train.py
    participant D as download.py
    participant PR as preprocess.py
    participant FE as features.py
    participant M as model (BaseModel)
    participant MF as MLflow

    S->>D: fetch_dataset()
    D-->>S: X_raw, y_raw (303 rows, 13 features)

    S->>PR: binarise_target(y_raw)
    PR-->>S: y_binary  (0 / 1)

    S->>PR: drop_missing(X_raw, y_binary)
    PR-->>S: X_clean, y_clean

    S->>PR: split(X_clean, y_clean)
    PR-->>S: X_train, X_test, y_train, y_test

    S->>MF: start_run(), log_params(...)

    S->>M: fit(X_train, y_train)
    M->>PR: build_preprocessing_pipeline() [inside model]
    M-->>S: fitted model

    S->>M: predict / predict_proba(X_test)
    M-->>S: y_pred, y_proba

    S->>MF: log_metrics(accuracy, f1, roc_auc)
    S->>M: save(models/<name>.pkl)
    S->>MF: log_model(artifact)
    MF-->>S: run_id
```

## API request lifecycle

```mermaid
sequenceDiagram
    participant C as Client
    participant FW as FastAPI
    participant V as Pydantic v2
    participant D as dependencies.py
    participant M as BaseModel
    participant R as PredictResponse

    C->>FW: POST /predict/ (JSON body)
    FW->>V: PredictRequest.model_validate(body)
    Note over V: Validates ranges, types<br>raises 422 on failure
    V-->>FW: PredictRequest (frozen)

    FW->>D: get_model(request)
    D-->>FW: model from app.state

    FW->>M: predict_proba(pd.DataFrame([features]))
    M->>M: preprocessor.transform(X)
    M-->>FW: ndarray [[p0, p1]]

    FW->>M: predict(pd.DataFrame([features]))
    M-->>FW: ndarray [label]

    FW->>R: PredictResponse(prediction, probability, model_version)
    R-->>C: 200 OK  { "prediction": 0, "probability": 0.12, "model_version": "1.0.0" }
```

## Key design decisions

| Decision | Rationale |
|----------|-----------|
| Pydantic v2 frozen models | Immutable request/response objects prevent accidental mutation downstream |
| Model loaded in lifespan hook | Errors surface at startup; the model is loaded exactly once rather than per-request |
| `app.state` for model injection | Allows the test suite to swap the model without patching globals |
| SQLAlchemy 2.0 async | Non-blocking DB writes do not add latency to the prediction path |
| Multi-stage Dockerfile | Builder stage installs deps; runtime stage ships only pre-built wheels and app source, keeping the image small |
| MLflow model registry | Enables versioning and stage promotion (staging -> production) across all four model types |
| BaseModel ABC | A single interface for all classifiers makes the training loop, evaluation, and API dependency generic and swappable |
