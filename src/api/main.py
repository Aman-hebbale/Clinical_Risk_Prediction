"""FastAPI application entry point."""

import logging
import os

import uvicorn
from fastapi import FastAPI

from src.api.routers.predict import router as predict_router
from src.api.schemas import HealthResponse, ModelInfoResponse
from src.models.registry import get_model_path

logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)

app: FastAPI = FastAPI(
    title="MedPredict",
    description="Heart disease prediction API using the UCI Heart Disease dataset.",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.include_router(predict_router)


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Liveness probe."""
    return HealthResponse(status="ok")


@app.get("/model/info", response_model=ModelInfoResponse, tags=["ops"])
def model_info() -> ModelInfoResponse:
    """Return metadata about the active model."""
    return ModelInfoResponse(
        model_path=str(get_model_path()),
        model_type="GradientBoostingClassifier",
        version="1.0.0",
    )


def start() -> None:
    """Entrypoint used by the ``medpredict-serve`` script."""
    host: str = os.getenv("HOST", "0.0.0.0")  # noqa: S104
    port: int = int(os.getenv("PORT", "8000"))
    uvicorn.run("src.api.main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    start()
