from pydantic import BaseModel, Field


class ClassProbability(BaseModel):
    label: str
    probability: float = Field(..., ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    label: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: list[ClassProbability]
    inference_time_ms: float


class BatchPredictionItem(BaseModel):
    """Result for one image in a batch. Either ``prediction`` or ``error`` is set."""

    filename: str | None = None
    success: bool
    prediction: PredictionResponse | None = None
    error: str | None = None


class BatchPredictionResponse(BaseModel):
    count: int
    results: list[BatchPredictionItem]


class ClassesResponse(BaseModel):
    classes: list[str]
    backbone: str
    image_size: int


class HealthResponse(BaseModel):
    status: str
    app_version: str
    model_loaded: bool
    device: str
    classes: list[str]


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    detail: str | None = None
