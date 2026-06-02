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

    loaded = False
    if settings.use_registry:
        try:
            import mlflow
            from mlflow.tracking import MlflowClient
            logger.info(f"Connecting to MLflow Tracking Server at {settings.mlflow_tracking_uri}...")
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            
            client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
            logger.info(f"Fetching champion version for registered model '{settings.mlflow_model_name}'...")
            mv = client.get_model_version_by_alias(settings.mlflow_model_name, "champion")
            logger.info(f"Resolved champion version v{mv.version} from source: {mv.source}")
            
            # Download the artifact file using direct source URI (works locally and remotely)
            download_path = mlflow.artifacts.download_artifacts(mv.source)
            model_file = download_path
            if os.path.isdir(download_path):
                for root, dirs, files in os.walk(download_path):
                    if "model.pt" in files:
                        model_file = os.path.join(root, "model.pt")
                        break
            
            logger.info(f"Champion model downloaded to: {model_file}")
            inference_service.load(model_file)
            loaded = True
        except Exception as e:
            logger.error(f"Failed to fetch champion model from MLflow Model Registry: {e}. Falling back to local files.")

    if not loaded:
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

# Prometheus instrumentation
try:
    from prometheus_fastapi_instrumentator import Instrumentator
    Instrumentator().instrument(app).expose(app)
    logger.info("Prometheus auto-instrumentation loaded and /metrics exposed")
except ImportError:
    from prometheus_client import make_asgi_app
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)
    logger.info("Prometheus client /metrics fallback mounted")


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
