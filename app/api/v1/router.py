from fastapi import APIRouter

from app.api.v1.endpoints import health, predict

v1_router = APIRouter(prefix="/api/v1")
v1_router.include_router(health.router)
v1_router.include_router(predict.router)
