from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration, sourced from environment variables / .env."""

    app_name: str = "PulmoScan AI"
    app_version: str = "1.0.0"
    env: str = "development"
    log_level: str = "INFO"

    # Path to the trained, self-describing checkpoint produced by the pipeline.
    model_path: str = "artifacts/training/model.pt"

    # Optional directory of fold checkpoints (model_fold*.pt). When it exists and
    # holds >=2 checkpoints, the API loads them as an ensemble (averaged softmax)
    # instead of the single model_path. Empty disables ensembling.
    ensemble_dir: str = "artifacts/training/folds"

    # Optional API-key gate. Leave empty to disable authentication.
    api_key: str = ""

    # CORS — comma-separated origins.
    cors_origins: str = "http://localhost:3000"

    # Reject uploads larger than this (bytes). Default 10 MB.
    max_upload_bytes: int = 10 * 1024 * 1024

    # Maximum number of images accepted in a single /predict/batch request.
    max_batch_size: int = 32

    # Test-time augmentation at inference (averages over a horizontal flip).
    # More accurate, ~2x inference cost. On — the free accuracy lever.
    use_tta: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


settings = Settings()
