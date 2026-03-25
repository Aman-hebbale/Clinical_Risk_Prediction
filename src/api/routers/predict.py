"""Prediction router."""

import logging
from typing import Annotated

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status

from src.api.dependencies import get_model
from src.api.schemas import PredictRequest, PredictResponse
from src.models.base import BaseModel as PredictModel

logger: logging.Logger = logging.getLogger(__name__)
router: APIRouter = APIRouter(prefix="/predict", tags=["prediction"])

MODEL_VERSION: str = "1.0.0"


@router.post("/", response_model=PredictResponse, status_code=status.HTTP_200_OK)
def predict(
    request: PredictRequest,
    model: Annotated[PredictModel, Depends(get_model)],
) -> PredictResponse:
    """Return a binary heart disease prediction with probability.

    Parameters
    ----------
    request:
        Patient feature values.
    model:
        Injected model instance.

    Returns
    -------
    PredictResponse
        Prediction and probability.
    """
    features: pd.DataFrame = pd.DataFrame([request.model_dump()])
    try:
        proba: float = float(model.predict_proba(features)[0, 1])
        label: int = int(model.predict(features)[0])
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed. Check server logs.",
        ) from exc

    return PredictResponse(
        prediction=label,
        probability=round(proba, 4),
        model_version=MODEL_VERSION,
    )
