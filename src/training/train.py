"""Main training script: fetch data, train all baselines, log to MLflow."""

import logging
import os
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.data.download import fetch_dataset
from src.data.preprocess import binarise_target, drop_missing, split
from src.models.base import BaseModel
from src.models.logistic_model import LogisticModel
from src.models.random_forest_model import RandomForestModel
from src.models.sklearn_model import SklearnModel

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)

MODEL_OUTPUT_DIR: Path = Path(os.getenv("MODEL_PATH", "models/best_model.pkl")).parent

# ---------------------------------------------------------------------------
# Baseline definitions
# ---------------------------------------------------------------------------

BASELINES: list[dict[str, Any]] = [
    {
        "name": "logistic_regression",
        "model_type": "LogisticRegression",
        "model_instance": LogisticModel(C=1.0, max_iter=1000),
        "params": {"C": 1.0, "max_iter": 1000},
    },
    {
        "name": "random_forest",
        "model_type": "RandomForestClassifier",
        "model_instance": RandomForestModel(n_estimators=200, max_depth=None),
        "params": {"n_estimators": 200, "max_depth": "None"},
    },
    {
        "name": "gradient_boosting",
        "model_type": "GradientBoostingClassifier",
        "model_instance": SklearnModel(
            n_estimators=200, learning_rate=0.05, max_depth=3
        ),
        "params": {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 3},
    },
]


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------


def evaluate(
    model: BaseModel,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Compute classification metrics on the test set.

    Parameters
    ----------
    model:
        Any trained model that satisfies the ``BaseModel`` interface.
    X_test:
        Test features.
    y_test:
        True binary labels.

    Returns
    -------
    dict[str, float]
        Mapping of metric name to value.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Fetch data, train all baselines, evaluate, log to MLflow, print table."""
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    experiment_name = "heart-disease-prediction"
    mlflow.set_experiment(experiment_name)

    logger.info("Fetching dataset ...")
    X_raw, y_raw = fetch_dataset()
    y_binary = binarise_target(y_raw)
    X_clean, y_clean = drop_missing(X_raw, y_binary)
    X_train, X_test, y_train, y_test = split(X_clean, y_clean)

    run_ids: list[str] = []

    for baseline in BASELINES:
        model_type: str = baseline["model_type"]
        model: BaseModel = baseline["model_instance"]
        params: dict[str, Any] = baseline["params"]
        artifact_path = f"model_{baseline['name']}"
        output_path: Path = MODEL_OUTPUT_DIR / f"{baseline['name']}.pkl"

        logger.info("Training %s ...", model_type)

        with mlflow.start_run() as run:
            mlflow.set_tags(
                {
                    "model_type": model_type,
                    "dataset_version": "uci-heart-disease",
                }
            )
            mlflow.log_params(params)

            model.fit(X_train, y_train)

            metrics = evaluate(model, X_test, y_test)
            mlflow.log_metrics(metrics)
            logger.info("%s metrics: %s", model_type, metrics)

            model.save(output_path)
            mlflow.sklearn.log_model(
                model._pipeline,  # noqa: SLF001
                artifact_path=artifact_path,
            )
            logger.info("Model artifact saved to %s", output_path)

            run_ids.append(run.info.run_id)

    # -----------------------------------------------------------------------
    # Comparison table
    # -----------------------------------------------------------------------
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        logger.error("Experiment '%s' not found in MLflow.", experiment_name)
        return

    runs_df: pd.DataFrame = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"run_id IN ({', '.join(repr(r) for r in run_ids)})",
    )

    if runs_df.empty:
        logger.warning("No runs found for the current training session.")
        return

    display_cols: list[str] = [
        "tags.model_type",
        "metrics.roc_auc",
        "metrics.accuracy",
        "metrics.f1",
    ]
    available = [c for c in display_cols if c in runs_df.columns]
    table = (
        runs_df[available]
        .rename(
            columns={
                "tags.model_type": "model_type",
                "metrics.roc_auc": "roc_auc",
                "metrics.accuracy": "accuracy",
                "metrics.f1": "f1",
            }
        )
        .sort_values("roc_auc", ascending=False)
        .reset_index(drop=True)
    )

    print("\n--- Baseline comparison ---")
    print(table.to_string(index=False, float_format="{:.4f}".format))
    best_row = table.iloc[0]
    print(f"\nBest model by AUC-ROC: {best_row['model_type']} "
          f"(roc_auc={best_row['roc_auc']:.4f})")


if __name__ == "__main__":
    main()
