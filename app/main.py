import glob
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from app.api.v1.router import v1_router
from app.config import settings
from app.core.logging import setup_logging
from app.core.middleware import RequestLoggingMiddleware
from app.services.inference import inference_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    logger.info(f"Starting {settings.app_name} v{settings.app_version} ({settings.env})")
    inference_service.use_tta = settings.use_tta
    # Prefer an ensemble of fold checkpoints when available; fall back to the
    # single model_path. Stay up in degraded mode if neither loads.
    fold_paths = (
        sorted(glob.glob(os.path.join(settings.ensemble_dir, "model_fold*.pt")))
        if settings.ensemble_dir
        else []
    )
    try:
        if len(fold_paths) >= 2:
            inference_service.load_ensemble(fold_paths)
        elif os.path.exists(settings.model_path):
            inference_service.load(settings.model_path)
        else:
            logger.warning(
                f"No model found (ensemble_dir={settings.ensemble_dir!r}, "
                f"model_path={settings.model_path}). API runs in degraded mode."
            )
    except Exception as e:  # noqa: BLE001 — keep API up in degraded mode
        logger.error(f"Failed to load model(s): {e}")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title=settings.app_name,
    description="Production API for chest CT-scan classification (PyTorch transfer-learning CNN).",
    version=settings.app_version,
    lifespan=lifespan,
)

app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    errors = [
        {"field": " -> ".join(str(part) for part in err["loc"]), "message": err["msg"]}
        for err in exc.errors()
    ]
    return JSONResponse(
        status_code=422, content={"success": False, "error": "Validation Error", "fields": errors}
    )


@app.exception_handler(Exception)
async def generic_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": "Internal Server Error"},
    )


app.include_router(v1_router)


@app.get("/", tags=["Root"])
def root() -> dict:
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/api/v1/health",
    }
