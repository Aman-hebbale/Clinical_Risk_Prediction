"""FastAPI dependency injection helpers."""

import logging
from functools import lru_cache

from src.models.base import BaseModel as PredictModel
from src.models.registry import load_model

logger: logging.Logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_model() -> PredictModel:
    """Return the singleton model instance (loaded once on first call).

    Raises
    ------
    RuntimeError
        If the model file cannot be found or loaded.
    """
    try:
        return load_model()
    except FileNotFoundError as exc:
        logger.error("Could not load model: %s", exc)
        raise RuntimeError(str(exc)) from exc
