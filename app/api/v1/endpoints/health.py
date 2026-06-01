from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.config import settings
from app.schemas.response import HealthResponse
from app.services.inference import inference_service

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Overall health: process is up and reports whether the model is loaded."""
    return HealthResponse(
        status="healthy" if inference_service.is_ready else "degraded",
        app_version=settings.app_version,
        model_loaded=inference_service.is_ready,
        device=inference_service.device,
        classes=inference_service.class_names,
    )


@router.get("/live")
def liveness() -> dict:
    """Liveness probe — the process is running."""
    return {"alive": True}


@router.get("/ready")
def readiness():
    """Readiness probe — returns 503 until the model is loaded."""
    if not inference_service.is_ready:
        return JSONResponse(
            status_code=503, content={"ready": False, "reason": "Model not loaded"}
        )
    return {"ready": True}
