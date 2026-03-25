"""Pydantic v2 request and response schemas for the prediction API."""

from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    """Input features for a single heart disease prediction."""

    model_config = ConfigDict(frozen=True)

    age: int = Field(..., ge=1, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="Sex (0 = female, 1 = male)")
    cp: int = Field(..., ge=0, le=3, description="Chest pain type (0-3)")
    trestbps: int = Field(..., ge=50, le=250, description="Resting blood pressure (mm Hg)")
    chol: int = Field(..., ge=100, le=600, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG results (0-2)")
    thalach: int = Field(..., ge=50, le=250, description="Maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise-induced angina (0/1)")
    oldpeak: float = Field(..., ge=0.0, le=10.0, description="ST depression induced by exercise")
    slope: int = Field(..., ge=0, le=2, description="Slope of peak exercise ST segment (0-2)")
    ca: int = Field(..., ge=0, le=4, description="Number of major vessels coloured by fluoroscopy")
    thal: int = Field(..., ge=0, le=3, description="Thalassemia type (0-3)")


class PredictResponse(BaseModel):
    """Prediction output returned to the caller."""

    model_config = ConfigDict(frozen=True)

    prediction: int = Field(..., description="Binary prediction (0 = no disease, 1 = disease)")
    probability: float = Field(..., ge=0.0, le=1.0, description="Probability of heart disease")
    model_version: str = Field(..., description="Version tag of the model that produced this result")


class HealthResponse(BaseModel):
    """Liveness probe response."""

    model_config = ConfigDict(frozen=True)

    status: str = "ok"


class ModelInfoResponse(BaseModel):
    """Metadata about the currently loaded model."""

    model_config = ConfigDict(frozen=True)

    model_path: str
    model_type: str
    version: str
