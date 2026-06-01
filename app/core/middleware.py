import time
import uuid

from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every request with a short request-id and a timing header."""

    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())[:8]
        start = time.time()

        response = await call_next(request)

        duration_ms = (time.time() - start) * 1000
        client = request.client.host if request.client else "unknown"
        logger.info(
            f"{request.method} {request.url.path} [{request_id}] "
            f"{response.status_code} {duration_ms:.1f}ms from {client}"
        )
        response.headers["X-Request-Id"] = request_id
        response.headers["X-Response-Time"] = f"{duration_ms:.1f}ms"
        return response
