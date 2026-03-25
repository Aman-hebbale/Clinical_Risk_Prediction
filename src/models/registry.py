"""Model registry: load the active model from disk or MLflow."""

import logging
import os
from pathlib import Path

from src.models.base import BaseModel
from src.models.sklearn_model import SklearnModel

logger: logging.Logger = logging.getLogger(__name__)

_DEFAULT_MODEL_PATH: Path = Path("models/best_model.pkl")


def get_model_path() -> Path:
    """Return the model path from the environment or fall back to the default."""
    raw: str = os.getenv("MODEL_PATH", str(_DEFAULT_MODEL_PATH))
    return Path(raw)


def load_model(path: Path | None = None) -> BaseModel:
    """Load the serialised model from *path*.

    Parameters
    ----------
    path:
        Path to the pickle file.  Defaults to ``MODEL_PATH`` env var.

    Returns
    -------
    BaseModel
        Loaded model instance.

    Raises
    ------
    FileNotFoundError
        If the model file does not exist.
    """
    resolved: Path = path or get_model_path()
    if not resolved.exists():
        raise FileNotFoundError(f"Model not found at {resolved}")
    logger.info("Loading model from %s", resolved)
    return SklearnModel.load(resolved)
