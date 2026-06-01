from pydantic import BaseModel, Field


class Base64ImageRequest(BaseModel):
    """Predict from a base64-encoded image payload."""

    image: str = Field(..., description="Base64-encoded image bytes (no data URI prefix).")

    model_config = {
        "json_schema_extra": {
            "examples": [{"image": "iVBORw0KGgoAAAANSUhEUgAA..."}]
        }
    }
