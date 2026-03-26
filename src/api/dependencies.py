"""FastAPI dependency injection helpers."""

import logging

from fastapi import Request

from src.models.base import BaseModel as PredictModel

logger: logging.Logger = logging.getLogger(__name__)


def get_model(request: Request) -> PredictModel:
    """Return the model instance stored in app state during startup.

    Raises
    ------
    RuntimeError
        If the model was not loaded at startup.
    """
    model: PredictModel | None = getattr(request.app.state, "model", None)
    if model is None:
        raise RuntimeError("Model is not loaded. Check startup logs.")
    return model
