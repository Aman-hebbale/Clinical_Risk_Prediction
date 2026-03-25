"""Main training script: fetch data, train model, log to MLflow."""

import logging
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.data.download import fetch_dataset
from src.data.preprocess import binarise_target, drop_missing, split
from src.models.sklearn_model import SklearnModel

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)

MODEL_OUTPUT_PATH: Path = Path(os.getenv("MODEL_PATH", "models/best_model.pkl"))


def evaluate(
    model: SklearnModel,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Compute classification metrics on the test set.

    Parameters
    ----------
    model:
        Trained model.
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


def main() -> None:
    """Fetch data, train, evaluate, log to MLflow, and persist the model."""
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("heart-disease-prediction")

    logger.info("Fetching dataset ...")
    X_raw, y_raw = fetch_dataset()
    y_binary = binarise_target(y_raw)
    X_clean, y_clean = drop_missing(X_raw, y_binary)
    X_train, X_test, y_train, y_test = split(X_clean, y_clean)

    params: dict[str, int | float] = {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 3,
    }

    with mlflow.start_run():
        mlflow.set_tags({"model_type": "GradientBoostingClassifier", "dataset": "uci-heart-disease"})
        mlflow.log_params(params)

        logger.info("Training model ...")
        model = SklearnModel(**params)  # type: ignore[arg-type]
        model.fit(X_train, y_train)

        metrics = evaluate(model, X_test, y_test)
        mlflow.log_metrics(metrics)
        logger.info("Metrics: %s", metrics)

        model.save(MODEL_OUTPUT_PATH)
        mlflow.sklearn.log_model(model._pipeline, artifact_path="model")  # noqa: SLF001
        logger.info("Model saved to %s", MODEL_OUTPUT_PATH)


if __name__ == "__main__":
    main()
