import sys

from loguru import logger

from app.config import settings


def setup_logging() -> None:
    """Configure loguru with a console sink and a rotating file sink."""
    logger.remove()
    logger.add(
        sys.stdout,
        level=settings.log_level,
        colorize=True,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
    )
    logger.add(
        "logs/api_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="14 days",
        compression="gz",
        level="INFO",
        format="{time} | {level} | {name}:{line} | {message}",
    )
