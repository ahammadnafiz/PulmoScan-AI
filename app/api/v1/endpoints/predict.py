import base64

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.config import settings
from app.core.security import require_api_key
from app.schemas.request import Base64ImageRequest
from app.schemas.response import ClassesResponse, PredictionResponse
from app.services.inference import inference_service

router = APIRouter(prefix="/predict", tags=["Prediction"], dependencies=[Depends(require_api_key)])

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}


def _ensure_ready() -> None:
    if not inference_service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded yet. Try again shortly.",
        )


def _run(image_bytes: bytes) -> PredictionResponse:
    try:
        result = inference_service.predict(image_bytes)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e)
        ) from e
    return PredictionResponse(**result)


@router.get("/classes", response_model=ClassesResponse)
def get_classes() -> ClassesResponse:
    """Return the classes the loaded model can predict."""
    _ensure_ready()
    return ClassesResponse(
        classes=inference_service.class_names,
        backbone=inference_service.backbone,
        image_size=inference_service.image_size,
    )


@router.post("", response_model=PredictionResponse, summary="Classify an uploaded image")
async def predict(file: UploadFile = File(..., description="CT scan image")) -> PredictionResponse:
    """Classify a chest CT-scan image uploaded as multipart form-data."""
    _ensure_ready()
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported content type '{file.content_type}'. Allowed: {sorted(ALLOWED_CONTENT_TYPES)}",
        )
    content = await file.read()
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds {settings.max_upload_bytes} bytes.",
        )
    return _run(content)


@router.post("/base64", response_model=PredictionResponse, summary="Classify a base64 image")
def predict_base64(request: Base64ImageRequest) -> PredictionResponse:
    """Classify an image sent as a base64-encoded JSON payload."""
    _ensure_ready()
    try:
        image_bytes = base64.b64decode(request.image)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid base64 payload: {e}",
        ) from e
    return _run(image_bytes)
