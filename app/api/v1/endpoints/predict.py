import base64

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.config import settings
from app.core.security import require_api_key
from app.schemas.request import Base64ImageRequest
from app.schemas.response import (
    BatchPredictionItem,
    BatchPredictionResponse,
    ClassesResponse,
    PredictionResponse,
)
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


def _validate_upload(content_type: str | None, size: int) -> None:
    """Validate one upload's content type and size. Raises ValueError on rejection."""
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise ValueError(
            f"Unsupported content type '{content_type}'. Allowed: {sorted(ALLOWED_CONTENT_TYPES)}"
        )
    if size > settings.max_upload_bytes:
        raise ValueError(f"File exceeds {settings.max_upload_bytes} bytes.")


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


@router.post(
    "/batch",
    response_model=BatchPredictionResponse,
    summary="Classify a batch of uploaded images",
)
async def predict_batch(
    files: list[UploadFile] = File(..., description="CT scan images"),
) -> BatchPredictionResponse:
    """Classify several chest CT-scan images in one multipart request.

    Each image is processed independently: a bad or oversized file produces an
    error for that item only, leaving the rest of the batch unaffected.
    """
    _ensure_ready()
    if not files:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No files provided.",
        )
    if len(files) > settings.max_batch_size:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Batch size {len(files)} exceeds limit of {settings.max_batch_size}.",
        )

    results: list[BatchPredictionItem] = []
    for file in files:
        content = await file.read()
        try:
            _validate_upload(file.content_type, len(content))
            prediction = PredictionResponse(**inference_service.predict(content))
        except ValueError as e:
            results.append(
                BatchPredictionItem(filename=file.filename, success=False, error=str(e))
            )
        else:
            results.append(
                BatchPredictionItem(
                    filename=file.filename, success=True, prediction=prediction
                )
            )

    return BatchPredictionResponse(count=len(results), results=results)


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
