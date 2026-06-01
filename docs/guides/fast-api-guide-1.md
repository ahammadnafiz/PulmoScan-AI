# 🚀 FastAPI for ML/AI Deployment — Complete Learning Guide
### Learn by Doing: From Zero to Production-Ready ML APIs

> **How to use this guide:** Read the theory, understand the *why*, then type (don't copy-paste) every code block. Each section builds on the last. By the end, you'll have a fully deployed ML API project.

---

## Table of Contents

1. [Why FastAPI for ML?](#1-why-fastapi-for-ml)
2. [Environment Setup](#2-environment-setup)
3. [FastAPI Core Concepts](#3-fastapi-core-concepts)
4. [Pydantic — Your Data Superpower](#4-pydantic--your-data-superpower)
5. [Building Your First ML Endpoint](#5-building-your-first-ml-endpoint)
6. [Request & Response Design Patterns](#6-request--response-design-patterns)
7. [Model Loading Strategies](#7-model-loading-strategies)
8. [Async, Background Tasks & Concurrency](#8-async-background-tasks--concurrency)
9. [File Uploads for ML (Images, CSVs)](#9-file-uploads-for-ml-images-csvs)
10. [Authentication & API Keys](#10-authentication--api-keys)
11. [Middleware, CORS & Rate Limiting](#11-middleware-cors--rate-limiting)
12. [Error Handling & Logging](#12-error-handling--logging)
13. [Testing Your ML API](#13-testing-your-ml-api)
14. [Docker & Deployment](#14-docker--deployment)
15. [🏗️ THE PROJECT: End-to-End ML Platform](#15-the-project-end-to-end-ml-platform)

---

## 1. Why FastAPI for ML?

### The Problem with Other Frameworks

When deploying ML models, you have unique needs:
- Handle large model files without crashing
- Process requests **concurrently** (inference is slow)
- Validate complex inputs (arrays, tensors, image data)
- Auto-document your API for team members
- Handle both sync and async model inference

**Flask** is synchronous — one request blocks another. **Django** is too heavy. **FastAPI** was built for exactly this.

### Why FastAPI Wins for ML

| Feature | Flask | Django | FastAPI |
|--------|-------|--------|---------|
| Async Support | ❌ | Partial | ✅ Native |
| Auto Docs (Swagger) | Manual | Manual | ✅ Auto |
| Data Validation | Manual | Forms only | ✅ Pydantic |
| Performance | Medium | Slow | ✅ Near Node.js |
| Type Safety | ❌ | ❌ | ✅ Full |
| Startup Time | Fast | Slow | Fast |

### The Mental Model

```
Client Request
    ↓
FastAPI Router  ← (matches URL + HTTP method)
    ↓
Pydantic        ← (validates & parses request data)
    ↓
Your Function   ← (run inference, business logic)
    ↓
Pydantic        ← (validates & serializes response)
    ↓
Client Response
```

---

## 2. Environment Setup

### Why Use Virtual Environments?

ML projects have conflicting dependencies (TensorFlow vs PyTorch versions). Virtual environments isolate them.

```bash
# Create project structure
mkdir ml-api-project
cd ml-api-project

# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# Verify you're inside venv
which python   # Should point to venv/python
```

### Install Dependencies

```bash
pip install fastapi==0.110.0
pip install "uvicorn[standard]==0.27.1"   # ASGI server
pip install pydantic==2.6.0
pip install python-multipart              # file uploads
pip install httpx                         # async HTTP client (for testing)
pip install pytest pytest-asyncio        # testing
pip install scikit-learn numpy pandas    # ML basics
pip install python-jose[cryptography]    # JWT auth
pip install passlib[bcrypt]              # password hashing
pip install slowapi                      # rate limiting
pip install loguru                       # better logging
pip install joblib                       # model serialization

# Save dependencies
pip freeze > requirements.txt
```

### Project Structure (learn this — it scales)

```
ml-api-project/
│
├── app/
│   ├── __init__.py
│   ├── main.py                  ← FastAPI app entry point
│   ├── config.py                ← Settings & env vars
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── v1/
│   │   │   ├── __init__.py
│   │   │   ├── router.py        ← Combines all v1 routes
│   │   │   └── endpoints/
│   │   │       ├── predict.py   ← Prediction endpoints
│   │   │       ├── health.py    ← Health check
│   │   │       └── auth.py      ← Authentication
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── security.py          ← Auth logic
│   │   └── middleware.py        ← Custom middleware
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── request_models.py    ← Pydantic input schemas
│   │   └── response_models.py   ← Pydantic output schemas
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   └── ml_service.py        ← ML model logic
│   │
│   └── ml_models/               ← Saved model files
│       └── .gitkeep
│
├── tests/
│   ├── __init__.py
│   ├── test_predict.py
│   └── test_auth.py
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

```bash
# Create the structure
mkdir -p app/api/v1/endpoints app/core app/models app/services app/ml_models tests
touch app/__init__.py app/main.py app/config.py
touch app/api/__init__.py app/api/v1/__init__.py app/api/v1/router.py
touch app/api/v1/endpoints/__init__.py app/api/v1/endpoints/predict.py
touch app/api/v1/endpoints/health.py app/api/v1/endpoints/auth.py
touch app/core/__init__.py app/core/security.py app/core/middleware.py
touch app/models/__init__.py app/models/request_models.py app/models/response_models.py
touch app/services/__init__.py app/services/ml_service.py
touch tests/__init__.py tests/test_predict.py
```

---

## 3. FastAPI Core Concepts

### 3.1 Your First App — Understanding Every Line

Create `app/main.py`:

```python
# app/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# FastAPI() creates the application instance
# Think of it as the "app object" like Flask's app = Flask(__name__)
app = FastAPI(
    title="ML Deployment API",          # Shows in /docs
    description="API for ML model inference",
    version="1.0.0",
    docs_url="/docs",                   # Swagger UI location
    redoc_url="/redoc",                 # ReDoc UI location
    openapi_url="/openapi.json"         # Raw OpenAPI schema
)

# Root endpoint — always have this for health checks
@app.get("/")
def root():
    return {"message": "ML API is running", "status": "healthy"}
```

Run it:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- `app.main` = the file path (`app/main.py`)
- `:app` = the FastAPI instance variable name
- `--reload` = auto-restart on code changes (dev only!)
- `--host 0.0.0.0` = accessible from other machines

Visit:
- `http://localhost:8000/` — your API
- `http://localhost:8000/docs` — **interactive Swagger docs (FREE!)**
- `http://localhost:8000/redoc` — alternative docs

### 3.2 HTTP Methods — When to Use What

```python
# GET  — Retrieve data, no body, idempotent
@app.get("/models")
def list_models():
    return {"models": ["sentiment", "classifier", "regressor"]}

# POST — Send data, create/process something
@app.post("/predict")
def predict(data: dict):
    return {"prediction": "positive"}

# PUT  — Update/replace something entirely  
@app.put("/models/{model_id}")
def update_model(model_id: str, config: dict):
    return {"updated": model_id}

# DELETE — Remove something
@app.delete("/models/{model_id}")
def delete_model(model_id: str):
    return {"deleted": model_id}

# PATCH — Partial update
@app.patch("/models/{model_id}")
def patch_model(model_id: str):
    return {"patched": model_id}
```

**ML API Rule of Thumb:**
- `GET /models` — list available models
- `POST /predict` — run inference (has request body)
- `GET /health` — health check
- `POST /train` — trigger training job
- `GET /results/{job_id}` — get async job result

### 3.3 Path Parameters vs Query Parameters vs Request Body

```python
from fastapi import Query, Path
from typing import Optional

# PATH PARAMETER — part of the URL, always required
# Use for: identifying a specific resource
@app.get("/models/{model_name}/predict")
def predict_with_model(
    model_name: str = Path(..., description="Name of the ML model to use")
):
    return {"model": model_name}

# QUERY PARAMETER — after ?, optional/required
# Use for: filtering, pagination, options
@app.get("/predictions")
def list_predictions(
    limit: int = Query(default=10, ge=1, le=100, description="Number of results"),
    offset: int = Query(default=0, ge=0),
    model_name: Optional[str] = Query(default=None)
):
    return {"limit": limit, "offset": offset, "model": model_name}

# REQUEST BODY — JSON payload
# Use for: input data for inference, complex objects
from pydantic import BaseModel

class PredictRequest(BaseModel):
    features: list[float]
    model_name: str = "default"

@app.post("/predict")
def predict(request: PredictRequest):
    # request.features, request.model_name are typed and validated
    return {"received_features": len(request.features)}
```

**URL Examples:**
- `/models/sentiment/predict` → path param: `model_name="sentiment"`
- `/predictions?limit=5&model_name=bert` → query params
- `POST /predict` with body `{"features": [1.2, 3.4]}` → body

### 3.4 Status Codes — Use Them Correctly

```python
from fastapi import status

@app.post("/predict", status_code=status.HTTP_200_OK)  # 200 OK (default)
def predict():
    return {"result": "..."}

@app.post("/jobs", status_code=status.HTTP_202_ACCEPTED)  # 202 for async jobs
def start_training():
    return {"job_id": "abc123", "status": "queued"}

@app.post("/models", status_code=status.HTTP_201_CREATED)  # 201 when created
def register_model():
    return {"model_id": "new_model"}
```

---

## 4. Pydantic — Your Data Superpower

### Why Pydantic?

When a client sends `{"age": "twenty"}` but you need an integer, Pydantic catches it **before** it reaches your ML model. No more `ValueError` crashes mid-inference.

### 4.1 Basic Models

Create `app/models/request_models.py`:

```python
# app/models/request_models.py

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, List, Literal
import numpy as np

# ─── Simple Example ───────────────────────────────────────────────
class SimpleTextRequest(BaseModel):
    text: str
    language: str = "en"  # Default value

# ─── ML Feature Vector ────────────────────────────────────────────
class FeatureVectorRequest(BaseModel):
    # Field() adds metadata: validation, docs, examples
    features: List[float] = Field(
        ...,                          # ... means REQUIRED
        min_length=1,
        max_length=1000,
        description="Numeric feature vector for model input",
        examples=[[1.2, 3.4, 5.6, 2.1]]
    )
    model_name: str = Field(
        default="default",
        description="Which model to use for inference"
    )
    return_probabilities: bool = Field(
        default=False,
        description="Return class probabilities instead of just the label"
    )

    # Custom validator on a single field
    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v):
        allowed = ["default", "v2", "experimental"]
        if v not in allowed:
            raise ValueError(f"model_name must be one of {allowed}")
        return v

    # Custom validator on the whole model (cross-field validation)
    @model_validator(mode="after")
    def check_experimental_features(self):
        if self.model_name == "experimental" and not self.return_probabilities:
            # Force probabilities for experimental model
            self.return_probabilities = True
        return self

# ─── NLP Request ──────────────────────────────────────────────────
class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    include_confidence: bool = True
    model_version: Literal["v1", "v2", "v3"] = "v2"  # Exact string choices

    @field_validator("text")
    @classmethod
    def clean_text(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("Text cannot be empty after stripping whitespace")
        return v

# ─── Batch Prediction ─────────────────────────────────────────────
class BatchPredictRequest(BaseModel):
    items: List[SentimentRequest] = Field(..., min_length=1, max_length=100)
    
    # Example for docs
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "items": [
                        {"text": "I love this product!", "model_version": "v2"},
                        {"text": "This is terrible.", "model_version": "v2"}
                    ]
                }
            ]
        }
    }

# ─── Image Classification ─────────────────────────────────────────
class ImageMetadata(BaseModel):
    width: int = Field(..., gt=0, le=10000)
    height: int = Field(..., gt=0, le=10000)
    channels: int = Field(default=3, ge=1, le=4)
    format: Literal["RGB", "BGR", "GRAY"] = "RGB"
```

### 4.2 Response Models

Create `app/models/response_models.py`:

```python
# app/models/response_models.py

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid

# ─── Standard API Response Wrapper ───────────────────────────────
# Always wrap responses — makes frontend integration easier
class APIResponse(BaseModel):
    success: bool = True
    message: str = "OK"
    data: Optional[Any] = None
    error: Optional[str] = None
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# ─── Prediction Responses ─────────────────────────────────────────
class PredictionResult(BaseModel):
    label: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: Optional[Dict[str, float]] = None

class SentimentResponse(BaseModel):
    text: str
    sentiment: str  # "positive", "negative", "neutral"
    confidence: float
    model_version: str
    inference_time_ms: float

class BatchPredictionResponse(BaseModel):
    results: List[SentimentResponse]
    total_items: int
    successful: int
    failed: int
    total_inference_time_ms: float

# ─── Model Info ───────────────────────────────────────────────────
class ModelInfo(BaseModel):
    name: str
    version: str
    description: str
    input_shape: Optional[List[int]] = None
    classes: Optional[List[str]] = None
    accuracy: Optional[float] = None
    is_loaded: bool

# ─── Job/Async Response ───────────────────────────────────────────
class JobResponse(BaseModel):
    job_id: str
    status: str  # "queued", "running", "completed", "failed"
    created_at: datetime
    result_url: Optional[str] = None  # where to fetch results

# ─── Health Check ─────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str  # "healthy", "degraded", "unhealthy"
    api_version: str
    models_loaded: List[str]
    uptime_seconds: float
    memory_usage_mb: float
```

### 4.3 Why Response Models Matter for Security

```python
# ❌ DANGEROUS — returns everything including internal fields
@app.get("/user/{id}")
def get_user(id: int):
    user = db.get_user(id)  # includes password_hash, internal_id, etc.
    return user  # NEVER DO THIS

# ✅ SAFE — Pydantic filters to only declared fields
class UserPublic(BaseModel):
    id: int
    username: str
    email: str
    # password_hash is NOT here, so it never leaves the server

@app.get("/user/{id}", response_model=UserPublic)
def get_user(id: int):
    user = db.get_user(id)
    return user  # FastAPI automatically filters using UserPublic
```

---

## 5. Building Your First ML Endpoint

### 5.1 Train & Save a Model (Run Once)

Create `app/services/train_sample_model.py`:

```python
# Run this script ONCE to create the model files
# python -m app.services.train_sample_model

import joblib
import numpy as np
from sklearn.datasets import load_iris, load_boston
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import os

def train_classifier():
    """Train a simple Iris classifier."""
    data = load_iris()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Pipeline: scale first, then classify
    # Why Pipeline? So scaling is part of the model — no data leakage
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    accuracy = pipeline.score(X_test, y_test)
    print(f"Classifier Accuracy: {accuracy:.4f}")

    # Save model
    os.makedirs("app/ml_models", exist_ok=True)
    joblib.dump(pipeline, "app/ml_models/iris_classifier.pkl")
    
    # Save metadata separately — useful for /models endpoint
    import json
    metadata = {
        "name": "iris_classifier",
        "version": "1.0.0",
        "description": "Iris flower species classifier",
        "classes": list(data.target_names),
        "input_features": list(data.feature_names),
        "accuracy": round(accuracy, 4)
    }
    with open("app/ml_models/iris_classifier_meta.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("Model saved to app/ml_models/iris_classifier.pkl")

if __name__ == "__main__":
    train_classifier()
```

```bash
python -m app.services.train_sample_model
```

### 5.2 The ML Service Layer

Create `app/services/ml_service.py`:

```python
# app/services/ml_service.py

"""
WHY A SERVICE LAYER?
- Keeps endpoints thin (just HTTP concerns)
- ML logic is reusable and testable separately
- Easy to swap model backends (sklearn → PyTorch)
"""

import joblib
import json
import time
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List
from loguru import logger

MODELS_DIR = Path("app/ml_models")

class ModelRegistry:
    """
    Manages loading and caching of ML models.
    
    WHY CACHE MODELS?
    Loading a model from disk takes 0.5-30 seconds.
    If you load it on every request, your API becomes unusable.
    Load once at startup, cache in memory, serve forever.
    """
    
    def __init__(self):
        self._models: Dict[str, Any] = {}
        self._metadata: Dict[str, Dict] = {}
        self._load_times: Dict[str, float] = {}
    
    def load_model(self, model_name: str) -> bool:
        """Load a model from disk into memory."""
        model_path = MODELS_DIR / f"{model_name}.pkl"
        meta_path = MODELS_DIR / f"{model_name}_meta.json"
        
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            return False
        
        try:
            start = time.time()
            self._models[model_name] = joblib.load(model_path)
            self._load_times[model_name] = time.time() - start
            
            if meta_path.exists():
                with open(meta_path) as f:
                    self._metadata[model_name] = json.load(f)
            
            logger.info(f"Loaded model '{model_name}' in {self._load_times[model_name]:.2f}s")
            return True
        except Exception as e:
            logger.error(f"Failed to load model '{model_name}': {e}")
            return False
    
    def get_model(self, model_name: str):
        """Retrieve a loaded model. Returns None if not loaded."""
        return self._models.get(model_name)
    
    def is_loaded(self, model_name: str) -> bool:
        return model_name in self._models
    
    def list_models(self) -> List[str]:
        return list(self._models.keys())
    
    def get_metadata(self, model_name: str) -> Optional[Dict]:
        return self._metadata.get(model_name)


# ─── Global Registry Instance ────────────────────────────────────
# Singleton pattern: one registry for the whole app
model_registry = ModelRegistry()


class PredictionService:
    """Handles inference logic for all models."""
    
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
    
    def predict_iris(
        self,
        features: List[float],
        return_probabilities: bool = False
    ) -> Dict[str, Any]:
        """
        Run inference on the iris classifier.
        
        Args:
            features: [sepal_length, sepal_width, petal_length, petal_width]
            return_probabilities: Whether to return class probabilities
            
        Returns:
            Prediction dict with label, confidence, optional probabilities
        """
        model = self.registry.get_model("iris_classifier")
        
        if model is None:
            raise RuntimeError("Model 'iris_classifier' is not loaded")
        
        # Validate feature count
        expected_features = 4
        if len(features) != expected_features:
            raise ValueError(
                f"Expected {expected_features} features, got {len(features)}"
            )
        
        # Convert to numpy array — sklearn needs 2D array (batch, features)
        X = np.array(features).reshape(1, -1)
        
        # Time the inference
        start = time.time()
        
        # Get prediction
        prediction = model.predict(X)[0]
        probas = model.predict_proba(X)[0]
        
        inference_time = (time.time() - start) * 1000  # Convert to ms
        
        # Map index to class name
        metadata = self.registry.get_metadata("iris_classifier")
        classes = metadata.get("classes", ["setosa", "versicolor", "virginica"])
        label = classes[prediction]
        confidence = float(probas[prediction])
        
        result = {
            "label": label,
            "confidence": round(confidence, 4),
            "inference_time_ms": round(inference_time, 3)
        }
        
        if return_probabilities:
            result["probabilities"] = {
                cls: round(float(prob), 4)
                for cls, prob in zip(classes, probas)
            }
        
        logger.debug(f"Prediction: {label} ({confidence:.2%}) in {inference_time:.1f}ms")
        return result
    
    def batch_predict_iris(
        self,
        batch: List[List[float]],
        return_probabilities: bool = False
    ) -> List[Dict]:
        """Efficient batch prediction — one forward pass for all items."""
        model = self.registry.get_model("iris_classifier")
        if model is None:
            raise RuntimeError("Model not loaded")
        
        metadata = self.registry.get_metadata("iris_classifier")
        classes = metadata.get("classes", ["setosa", "versicolor", "virginica"])
        
        # Stack all samples — efficient matrix operation
        X = np.array(batch)
        
        start = time.time()
        predictions = model.predict(X)
        probas = model.predict_proba(X)
        total_time = (time.time() - start) * 1000
        
        results = []
        for i, (pred, proba) in enumerate(zip(predictions, probas)):
            label = classes[pred]
            result = {
                "label": label,
                "confidence": round(float(proba[pred]), 4),
            }
            if return_probabilities:
                result["probabilities"] = {
                    cls: round(float(p), 4)
                    for cls, p in zip(classes, proba)
                }
            results.append(result)
        
        logger.info(f"Batch of {len(batch)} items predicted in {total_time:.1f}ms")
        return results


# ─── Service Instances ────────────────────────────────────────────
prediction_service = PredictionService(model_registry)
```

### 5.3 The Prediction Endpoint

Create `app/api/v1/endpoints/predict.py`:

```python
# app/api/v1/endpoints/predict.py

from fastapi import APIRouter, HTTPException, Depends, status
from typing import List
from loguru import logger
import time

from app.models.request_models import FeatureVectorRequest, BatchPredictRequest
from app.models.response_models import (
    APIResponse, PredictionResult, BatchPredictionResponse
)
from app.services.ml_service import prediction_service

router = APIRouter(prefix="/predict", tags=["Predictions"])


@router.post(
    "/iris",
    response_model=APIResponse,
    summary="Classify Iris Flower",
    description="Predicts the species of an iris flower from 4 measurements."
)
def predict_iris(request: FeatureVectorRequest):
    """
    Predict iris flower species.
    
    Features (in order):
    - sepal length (cm)
    - sepal width (cm)
    - petal length (cm)
    - petal width (cm)
    """
    try:
        result = prediction_service.predict_iris(
            features=request.features,
            return_probabilities=request.return_probabilities
        )
        
        return APIResponse(
            success=True,
            message="Prediction successful",
            data=result
        )
    
    except ValueError as e:
        # Client error — wrong input
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except RuntimeError as e:
        # Server error — model not loaded
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during inference"
        )


@router.post(
    "/iris/batch",
    response_model=APIResponse,
    summary="Batch Classify Iris Flowers"
)
def predict_iris_batch(
    items: List[FeatureVectorRequest],
    # WHY LIMIT BATCH SIZE? Protect server memory & fairness
):
    """Predict species for multiple samples in one call — much faster than N separate calls."""
    
    if len(items) > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch size cannot exceed 100 items"
        )
    
    batch = [item.features for item in items]
    return_proba = items[0].return_probabilities if items else False
    
    start = time.time()
    try:
        results = prediction_service.batch_predict_iris(batch, return_proba)
        total_ms = (time.time() - start) * 1000
        
        return APIResponse(
            success=True,
            message=f"Batch prediction complete: {len(results)} items",
            data={
                "results": results,
                "total_items": len(results),
                "total_inference_time_ms": round(total_ms, 2),
                "avg_inference_time_ms": round(total_ms / len(results), 2)
            }
        )
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

---

## 6. Request & Response Design Patterns

### 6.1 Dependency Injection — FastAPI's Superpower

**Why DI?** Avoids repeating code across endpoints. Auth, DB connections, model loading — all via DI.

```python
# app/core/dependencies.py

from fastapi import Depends, HTTPException, Header, status
from typing import Optional, Annotated
from app.services.ml_service import model_registry

# ─── Model Dependency ─────────────────────────────────────────────
def get_model(model_name: str = "iris_classifier"):
    """
    Dependency: ensures a model is loaded before the endpoint runs.
    FastAPI calls this function BEFORE your endpoint function.
    """
    if not model_registry.is_loaded(model_name):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model '{model_name}' is not available. Server may be starting up."
        )
    return model_registry.get_model(model_name)

# ─── Request ID ───────────────────────────────────────────────────
import uuid

def get_request_id(x_request_id: Optional[str] = Header(default=None)):
    """
    Extract or generate a request ID.
    Clients can send X-Request-Id header, or we generate one.
    Useful for tracing requests across logs.
    """
    return x_request_id or str(uuid.uuid4())

# ─── Usage in Endpoint ────────────────────────────────────────────
# from fastapi import Depends
#
# @router.post("/predict")
# def predict(
#     request: PredictRequest,
#     model = Depends(get_model),               ← FastAPI injects this
#     request_id: str = Depends(get_request_id) ← And this
# ):
#     ...
```

### 6.2 API Versioning — Design for the Future

```python
# app/api/v1/router.py
from fastapi import APIRouter
from app.api.v1.endpoints import predict, health, auth

v1_router = APIRouter(prefix="/api/v1")
v1_router.include_router(predict.router)
v1_router.include_router(health.router)
v1_router.include_router(auth.router)

# app/main.py
from app.api.v1.router import v1_router
app.include_router(v1_router)

# WHY VERSION? When you change your API (e.g., rename a field),
# v1 clients still work while v2 uses the new format.
# URL becomes: /api/v1/predict/iris
```

### 6.3 Health Check Endpoint

Create `app/api/v1/endpoints/health.py`:

```python
# app/api/v1/endpoints/health.py

from fastapi import APIRouter
from datetime import datetime
import psutil  # pip install psutil
import time
from app.services.ml_service import model_registry
from app.models.response_models import HealthResponse

router = APIRouter(prefix="/health", tags=["Health"])

# Track startup time
_start_time = time.time()

@router.get("/", response_model=HealthResponse)
def health_check():
    """
    Health check endpoint.
    Used by: load balancers, Kubernetes, Docker, monitoring tools.
    Returns 200 if healthy, 503 if not.
    """
    uptime = time.time() - _start_time
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    
    loaded_models = model_registry.list_models()
    
    # Determine health status
    status = "healthy"
    if not loaded_models:
        status = "degraded"  # App is up but no models loaded
    
    return HealthResponse(
        status=status,
        api_version="1.0.0",
        models_loaded=loaded_models,
        uptime_seconds=round(uptime, 2),
        memory_usage_mb=round(memory_mb, 2)
    )

@router.get("/ready")
def readiness_check():
    """
    Kubernetes readiness probe.
    Returns 200 only when app is ready to serve traffic.
    """
    if not model_registry.list_models():
        return {"ready": False, "reason": "No models loaded"}, 503
    return {"ready": True}

@router.get("/live")
def liveness_check():
    """
    Kubernetes liveness probe.
    Returns 200 if the process is alive (even if models aren't loaded).
    """
    return {"alive": True, "timestamp": datetime.utcnow().isoformat()}
```

---

## 7. Model Loading Strategies

### 7.1 Startup Events — Load Models at Boot

```python
# app/main.py — COMPLETE VERSION

from contextlib import asynccontextmanager
from fastapi import FastAPI
from loguru import logger
from app.services.ml_service import model_registry
from app.api.v1.router import v1_router

# ─── Lifespan Manager ─────────────────────────────────────────────
# WHY LIFESPAN? FastAPI's modern way to run code at startup/shutdown
# Replaces the old @app.on_event("startup") decorator
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ═══ STARTUP ════
    logger.info("🚀 Starting ML API...")
    
    # Load all models into memory
    models_to_load = ["iris_classifier"]
    for model_name in models_to_load:
        success = model_registry.load_model(model_name)
        if success:
            logger.info(f"✅ Model '{model_name}' loaded successfully")
        else:
            logger.warning(f"⚠️  Could not load model '{model_name}'")
    
    logger.info(f"✅ API ready. Loaded models: {model_registry.list_models()}")
    
    yield  # ← Application runs here
    
    # ═══ SHUTDOWN ════
    logger.info("Shutting down ML API...")
    # Clean up resources, close DB connections, etc.

app = FastAPI(
    title="ML Deployment API",
    description="""
    ## Machine Learning Model Serving API
    
    Deploy and query ML models via REST API.
    
    ### Features
    - 🔮 Real-time predictions
    - 📦 Batch processing
    - 🔐 API key authentication
    - 📊 Health monitoring
    """,
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(v1_router)

@app.get("/", tags=["Root"])
def root():
    return {
        "message": "ML API is running",
        "docs": "/docs",
        "health": "/api/v1/health"
    }
```

### 7.2 Lazy Loading vs Eager Loading

```python
# ─── EAGER LOADING (recommended for production) ───────────────────
# Load all models at startup
# PRO: No latency on first request
# CON: Longer startup time, more memory used upfront

# In lifespan() startup:
for model in ["model_a", "model_b", "model_c"]:
    model_registry.load_model(model)

# ─── LAZY LOADING (good for development/many models) ──────────────
# Load models on first request
# PRO: Faster startup, only loads what's needed
# CON: First request is slow, concurrency issues

from threading import Lock

_load_lock = Lock()  # Prevent multiple threads loading same model

def get_or_load_model(model_name: str):
    if not model_registry.is_loaded(model_name):
        with _load_lock:  # Thread-safe
            if not model_registry.is_loaded(model_name):  # Double check
                model_registry.load_model(model_name)
    return model_registry.get_model(model_name)
```

---

## 8. Async, Background Tasks & Concurrency

### 8.1 Sync vs Async — The Critical Decision for ML

```python
# SYNC endpoint — blocks the thread during execution
# Use when: calling blocking I/O, CPU-bound work (sklearn inference)
@router.post("/predict/sync")
def predict_sync(request: FeatureVectorRequest):
    # FastAPI runs this in a thread pool — doesn't block other requests
    result = prediction_service.predict_iris(request.features)
    return result

# ASYNC endpoint — non-blocking, uses event loop
# Use when: calling async databases, async HTTP clients, async ML APIs
@router.post("/predict/async")
async def predict_async(request: FeatureVectorRequest):
    # ⚠️  Don't run CPU-heavy work directly in async endpoints!
    # It blocks the event loop and hurts ALL other requests
    
    import asyncio
    loop = asyncio.get_event_loop()
    
    # Run CPU-heavy inference in thread pool executor (non-blocking)
    result = await loop.run_in_executor(
        None,  # None = default executor
        prediction_service.predict_iris,
        request.features
    )
    return result

# ─── THE RULE ─────────────────────────────────────────────────────
# sklearn/numpy/pandas inference → use sync def (FastAPI threads it)
# Calling external APIs, DBs, file I/O → use async def with await
# PyTorch/TF with CUDA → use sync def with thread pool
```

### 8.2 Background Tasks — Fire and Forget

```python
from fastapi import BackgroundTasks
import asyncio

def save_prediction_log(request_data: dict, result: dict):
    """This runs in the background after response is sent."""
    # Save to database, send metrics to monitoring system, etc.
    logger.info(f"Logging prediction: input={request_data}, output={result}")
    # db.insert_prediction_log(request_data, result)

@router.post("/predict/iris")
def predict_with_logging(
    request: FeatureVectorRequest,
    background_tasks: BackgroundTasks  # FastAPI injects this
):
    result = prediction_service.predict_iris(request.features)
    
    # Schedule background task — runs AFTER response is sent to client
    # Client gets response immediately, logging happens behind the scenes
    background_tasks.add_task(
        save_prediction_log,
        request_data=request.model_dump(),
        result=result
    )
    
    return APIResponse(success=True, data=result)

# ─── Long Running Async Job ───────────────────────────────────────
import uuid
from typing import Dict

# In-memory job store (use Redis in production)
job_store: Dict[str, dict] = {}

@router.post("/train", status_code=202)
async def start_training_job(background_tasks: BackgroundTasks):
    """
    Start a long-running training job.
    Returns job_id immediately, client polls for result.
    202 Accepted = "I got it, working on it."
    """
    job_id = str(uuid.uuid4())
    job_store[job_id] = {"status": "queued", "result": None}
    
    background_tasks.add_task(run_training_job, job_id)
    
    return {"job_id": job_id, "status": "queued", "poll_url": f"/api/v1/jobs/{job_id}"}

async def run_training_job(job_id: str):
    """Simulated long-running training."""
    job_store[job_id]["status"] = "running"
    await asyncio.sleep(10)  # Simulate training time
    job_store[job_id] = {"status": "completed", "result": {"accuracy": 0.95}}

@router.get("/jobs/{job_id}")
def get_job_status(job_id: str):
    if job_id not in job_store:
        raise HTTPException(status_code=404, detail="Job not found")
    return job_store[job_id]
```

---

## 9. File Uploads for ML (Images, CSVs)

### 9.1 Image Upload for Vision Models

```python
# app/api/v1/endpoints/files.py

from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from typing import Optional, List
import io
import numpy as np
from PIL import Image  # pip install Pillow
from loguru import logger

router = APIRouter(prefix="/files", tags=["File Upload"])

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

async def validate_image(file: UploadFile) -> bytes:
    """Validate and read uploaded image."""
    # Check content type
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid file type: {file.content_type}. Allowed: {ALLOWED_IMAGE_TYPES}"
        )
    
    # Read file content
    content = await file.read()
    
    # Check file size
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {len(content)/1024/1024:.1f}MB. Max: 10MB"
        )
    
    return content

def preprocess_image(image_bytes: bytes, target_size=(224, 224)) -> np.ndarray:
    """
    Standard image preprocessing pipeline.
    Mimics what torchvision.transforms or tf.keras.preprocessing does.
    """
    # Open image from bytes
    img = Image.open(io.BytesIO(image_bytes))
    
    # Convert to RGB (handle RGBA, grayscale, etc.)
    img = img.convert("RGB")
    
    # Resize to model's expected input
    img = img.resize(target_size, Image.LANCZOS)
    
    # Convert to numpy array
    arr = np.array(img, dtype=np.float32)
    
    # Normalize to [0, 1]
    arr = arr / 255.0
    
    # ImageNet normalization (for pretrained models)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    arr = (arr - mean) / std
    
    # Add batch dimension: (224, 224, 3) → (1, 224, 224, 3)
    arr = np.expand_dims(arr, axis=0)
    
    return arr

@router.post("/classify-image")
async def classify_image(
    file: UploadFile = File(..., description="Image file to classify"),
    model_name: str = Form(default="default", description="Model to use"),
    return_top_k: int = Form(default=5)
):
    """
    Upload an image and get classification results.
    
    Use multipart/form-data content type.
    """
    logger.info(f"Received image: {file.filename}, size: {file.size}, type: {file.content_type}")
    
    # Validate
    content = await validate_image(file)
    
    # Preprocess
    preprocessed = preprocess_image(content)
    logger.debug(f"Preprocessed shape: {preprocessed.shape}")
    
    # ─── Replace with actual model inference ───────────────────
    # model = model_registry.get_model(model_name)
    # predictions = model.predict(preprocessed)
    # For demo, return dummy result:
    dummy_result = {
        "filename": file.filename,
        "preprocessed_shape": list(preprocessed.shape),
        "predictions": [
            {"class": "cat", "confidence": 0.87},
            {"class": "dog", "confidence": 0.11},
            {"class": "fox", "confidence": 0.02}
        ][:return_top_k]
    }
    
    return {"success": True, "data": dummy_result}

@router.post("/batch-classify")
async def batch_classify(
    files: List[UploadFile] = File(..., description="Multiple images to classify")
):
    """Upload multiple images at once."""
    if len(files) > 20:
        raise HTTPException(status_code=400, detail="Max 20 files per batch")
    
    results = []
    for file in files:
        try:
            content = await validate_image(file)
            preprocessed = preprocess_image(content)
            results.append({
                "filename": file.filename,
                "status": "success",
                "shape": list(preprocessed.shape)
            })
        except HTTPException as e:
            results.append({
                "filename": file.filename,
                "status": "error",
                "error": e.detail
            })
    
    return {"results": results, "total": len(results)}

# ─── CSV Upload for Batch Predictions ─────────────────────────────
import pandas as pd

@router.post("/predict-csv")
async def predict_from_csv(
    file: UploadFile = File(..., description="CSV file with features"),
    model_name: str = Form(default="iris_classifier")
):
    """
    Upload a CSV file and get predictions for all rows.
    
    Expected CSV format: sepal_length,sepal_width,petal_length,petal_width
    """
    if file.content_type not in {"text/csv", "application/csv"}:
        raise HTTPException(status_code=422, detail="File must be CSV")
    
    content = await file.read()
    
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid CSV: {e}")
    
    # Validate columns
    required_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing columns: {missing}. Required: {required_cols}"
        )
    
    if len(df) > 10000:
        raise HTTPException(status_code=400, detail="CSV too large. Max 10,000 rows")
    
    # Get features
    X = df[required_cols].values.tolist()
    
    from app.services.ml_service import prediction_service
    results = prediction_service.batch_predict_iris(X)
    
    # Add predictions back to dataframe
    df["predicted_species"] = [r["label"] for r in results]
    df["confidence"] = [r["confidence"] for r in results]
    
    return {
        "success": True,
        "rows_processed": len(df),
        "predictions": df[required_cols + ["predicted_species", "confidence"]].to_dict(orient="records")
    }
```

---

## 10. Authentication & API Keys

### Why Authenticate Your ML API?

ML inference is expensive. Without auth:
- Anyone can spam your API
- Competitors can use your models for free
- No usage tracking per user

### 10.1 API Key Authentication

Create `app/core/security.py`:

```python
# app/core/security.py

from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader, APIKeyQuery
from typing import Optional

# ─── Simple API Key (good for most ML APIs) ───────────────────────
# Keys can be passed in header OR query param

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
API_KEY_QUERY = APIKeyQuery(name="api_key", auto_error=False)

# In production: store in database with user info, expiry, rate limits
VALID_API_KEYS = {
    "sk-dev-1234567890abcdef": {"user": "developer", "tier": "free"},
    "sk-prod-abcdefghij123456": {"user": "production_user", "tier": "pro"},
}

async def get_api_key(
    header_key: Optional[str] = Security(API_KEY_HEADER),
    query_key: Optional[str] = Security(API_KEY_QUERY),
):
    """
    Dependency: validates API key from header or query param.
    
    Usage:
        curl -H "X-API-Key: sk-dev-..." http://localhost:8000/predict
        OR
        curl http://localhost:8000/predict?api_key=sk-dev-...
    """
    api_key = header_key or query_key
    
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required. Pass X-API-Key header or api_key query parameter.",
            headers={"WWW-Authenticate": "ApiKey"}
        )
    
    if api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key"
        )
    
    return VALID_API_KEYS[api_key]  # Returns user info dict


# ─── JWT Token Authentication (for user-facing APIs) ──────────────
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel

SECRET_KEY = "your-secret-key-change-in-production-use-env-var"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token")

# Fake user DB — replace with real database
FAKE_USERS_DB = {
    "alice": {
        "username": "alice",
        "hashed_password": pwd_context.hash("secret123"),
        "tier": "pro"
    }
}

class Token(BaseModel):
    access_token: str
    token_type: str
    expires_in: int

class TokenData(BaseModel):
    username: Optional[str] = None

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Security(oauth2_scheme)):
    """Dependency: validates JWT and returns user."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = FAKE_USERS_DB.get(username)
    if user is None:
        raise credentials_exception
    return user
```

Create `app/api/v1/endpoints/auth.py`:

```python
# app/api/v1/endpoints/auth.py

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from app.core.security import (
    verify_password, create_access_token, Token, FAKE_USERS_DB
)
from datetime import timedelta

router = APIRouter(prefix="/auth", tags=["Authentication"])

@router.post("/token", response_model=Token)
async def login_for_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    OAuth2 login endpoint.
    
    Returns a JWT token for authenticated API access.
    
    Usage:
        curl -X POST /api/v1/auth/token \\
             -d "username=alice&password=secret123"
    """
    user = FAKE_USERS_DB.get(form_data.username)
    
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )
    
    token = create_access_token(
        data={"sub": user["username"]},
        expires_delta=timedelta(minutes=30)
    )
    
    return Token(
        access_token=token,
        token_type="bearer",
        expires_in=30 * 60  # seconds
    )

# Protected endpoint example
from app.core.security import get_current_user, get_api_key
from fastapi import Security

@router.get("/me")
async def get_current_user_info(
    current_user: dict = Depends(get_current_user)  # JWT auth
):
    return {"user": current_user["username"], "tier": current_user["tier"]}

@router.get("/me/apikey")
async def get_current_user_via_apikey(
    api_key_info: dict = Depends(get_api_key)  # API key auth
):
    return api_key_info
```

---

## 11. Middleware, CORS & Rate Limiting

### 11.1 CORS — Let Your Frontend Talk to the API

```python
# app/main.py — add to app setup

from fastapi.middleware.cors import CORSMiddleware

# CORS = Cross-Origin Resource Sharing
# WHY? Browsers block JS from calling APIs on different domains by default.
# Your React app on localhost:3000 can't call API on localhost:8000 without this.

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",     # Local React dev
        "https://yourdomain.com",    # Production frontend
        # "∗" — DON'T use in production! Any site can call your API
    ],
    allow_credentials=True,          # Allow cookies/auth headers
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    allow_headers=["*"],             # Allow all request headers
)
```

### 11.2 Rate Limiting — Protect Your Inference Budget

```python
# app/core/middleware.py

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request

# Create limiter — uses client IP by default
limiter = Limiter(key_func=get_remote_address)

# In app/main.py:
# from app.core.middleware import limiter
# app.state.limiter = limiter
# app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Usage in endpoints:
# @router.post("/predict")
# @limiter.limit("10/minute")  ← max 10 requests per minute per IP
# async def predict(request: Request, ...):
#     ...
```

### 11.3 Custom Middleware — Request Logging, Timing

```python
# app/core/middleware.py (add to existing)

import time
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from loguru import logger

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Logs every request with timing info.
    Runs before and after EVERY endpoint.
    """
    
    async def dispatch(self, request: Request, call_next):
        # ─── BEFORE endpoint ───────────────────────────────────
        request_id = str(uuid.uuid4())[:8]
        start_time = time.time()
        
        logger.info(
            f"→ {request.method} {request.url.path} "
            f"[{request_id}] from {request.client.host}"
        )
        
        # ─── Call the actual endpoint ──────────────────────────
        response: Response = await call_next(request)
        
        # ─── AFTER endpoint ────────────────────────────────────
        duration_ms = (time.time() - start_time) * 1000
        
        logger.info(
            f"← {request.method} {request.url.path} "
            f"[{request_id}] {response.status_code} "
            f"in {duration_ms:.1f}ms"
        )
        
        # Add headers to response
        response.headers["X-Request-Id"] = request_id
        response.headers["X-Response-Time"] = f"{duration_ms:.1f}ms"
        
        return response

# In app/main.py:
# from app.core.middleware import RequestLoggingMiddleware
# app.add_middleware(RequestLoggingMiddleware)
```

---

## 12. Error Handling & Logging

### 12.1 Global Exception Handlers

```python
# app/main.py — add exception handlers

from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

# ─── Pydantic Validation Error ────────────────────────────────────
# Triggered when request body doesn't match the Pydantic model
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Return friendly error messages for invalid input."""
    errors = []
    for error in exc.errors():
        errors.append({
            "field": " → ".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"]
        })
    
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": "Validation Error",
            "detail": "Request data is invalid",
            "fields": errors
        }
    )

# ─── Generic 500 Error ────────────────────────────────────────────
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal Server Error",
            "detail": "An unexpected error occurred. Please try again."
        }
    )
```

### 12.2 Loguru — Production-Grade Logging

```python
# app/config.py

from loguru import logger
import sys
import os

def setup_logging():
    """Configure loguru for the application."""
    
    # Remove default handler
    logger.remove()
    
    # Console output (development)
    logger.add(
        sys.stdout,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        level="DEBUG" if os.getenv("ENV", "dev") == "dev" else "INFO",
        colorize=True
    )
    
    # File output (production)
    logger.add(
        "logs/ml_api_{time:YYYY-MM-DD}.log",
        rotation="1 day",      # New file each day
        retention="30 days",   # Keep last 30 days
        compression="gz",      # Compress old logs
        format="{time} | {level} | {name}:{line} | {message}",
        level="INFO"
    )
    
    return logger
```

---

## 13. Testing Your ML API

### 13.1 Setting Up Tests

```python
# tests/conftest.py

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport
import asyncio
from app.main import app
from app.services.ml_service import model_registry

@pytest.fixture(scope="session")
def client():
    """Synchronous test client — use for most tests."""
    # Load models for testing
    model_registry.load_model("iris_classifier")
    with TestClient(app) as c:
        yield c

@pytest.fixture(scope="session")
async def async_client():
    """Async test client — use for async endpoints."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as c:
        yield c
```

```python
# tests/test_predict.py

import pytest
from fastapi.testclient import TestClient

class TestIrisPrediction:
    """Test suite for iris prediction endpoint."""
    
    def test_predict_valid_input(self, client: TestClient):
        """Happy path — valid features return a prediction."""
        response = client.post(
            "/api/v1/predict/iris",
            json={
                "features": [5.1, 3.5, 1.4, 0.2],  # Known setosa
                "return_probabilities": False
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "label" in data["data"]
        assert data["data"]["label"] in ["setosa", "versicolor", "virginica"]
        assert 0 <= data["data"]["confidence"] <= 1
    
    def test_predict_with_probabilities(self, client: TestClient):
        """Test that probabilities sum to ~1.0."""
        response = client.post(
            "/api/v1/predict/iris",
            json={"features": [5.1, 3.5, 1.4, 0.2], "return_probabilities": True}
        )
        assert response.status_code == 200
        probas = response.json()["data"]["probabilities"]
        assert probas is not None
        total = sum(probas.values())
        assert abs(total - 1.0) < 0.001  # Should sum to 1
    
    def test_predict_wrong_feature_count(self, client: TestClient):
        """Wrong number of features → 422 error."""
        response = client.post(
            "/api/v1/predict/iris",
            json={"features": [1.0, 2.0]}  # Only 2 features, need 4
        )
        assert response.status_code == 422
    
    def test_predict_invalid_model_name(self, client: TestClient):
        """Invalid model name → 422 validation error."""
        response = client.post(
            "/api/v1/predict/iris",
            json={"features": [5.1, 3.5, 1.4, 0.2], "model_name": "nonexistent"}
        )
        assert response.status_code == 422
    
    def test_predict_empty_features(self, client: TestClient):
        """Empty features list → 422."""
        response = client.post(
            "/api/v1/predict/iris",
            json={"features": []}
        )
        assert response.status_code == 422
    
    def test_batch_prediction(self, client: TestClient):
        """Batch should return same number of results as input."""
        samples = [
            {"features": [5.1, 3.5, 1.4, 0.2]},
            {"features": [6.3, 3.3, 6.0, 2.5]},
            {"features": [5.9, 3.0, 5.1, 1.8]},
        ]
        response = client.post("/api/v1/predict/iris/batch", json=samples)
        assert response.status_code == 200
        data = response.json()["data"]
        assert len(data["results"]) == 3

class TestHealth:
    def test_health_check(self, client: TestClient):
        response = client.get("/api/v1/health/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "degraded"]
        assert "models_loaded" in data
    
    def test_root_endpoint(self, client: TestClient):
        response = client.get("/")
        assert response.status_code == 200

class TestAuthentication:
    def test_protected_endpoint_without_key(self, client: TestClient):
        """Should return 401 without API key."""
        response = client.get("/api/v1/auth/me/apikey")
        assert response.status_code == 401  # or 403
    
    def test_protected_endpoint_with_valid_key(self, client: TestClient):
        response = client.get(
            "/api/v1/auth/me/apikey",
            headers={"X-API-Key": "sk-dev-1234567890abcdef"}
        )
        assert response.status_code == 200
```

Run tests:
```bash
pytest tests/ -v                    # Run all tests
pytest tests/ -v --tb=short        # Short traceback
pytest tests/test_predict.py -v    # Specific file
pytest -k "test_health"            # Tests matching pattern
pytest --cov=app tests/            # With coverage report (pip install pytest-cov)
```

---

## 14. Docker & Deployment

### 14.1 Dockerfile

```dockerfile
# Dockerfile

# ─── Base Image ────────────────────────────────────────────────────
# Use slim for smaller image size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# ─── Environment Variables ────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ENV=production \
    PORT=8000

# ─── Install Dependencies ─────────────────────────────────────────
# Copy requirements first (Docker layer caching)
# If requirements.txt doesn't change, this layer is cached
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ─── Copy Application ─────────────────────────────────────────────
COPY app/ ./app/

# ─── Create logs directory ────────────────────────────────────────
RUN mkdir -p logs

# ─── Non-root User (security) ─────────────────────────────────────
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# ─── Expose Port ──────────────────────────────────────────────────
EXPOSE 8000

# ─── Health Check ─────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health/live || exit 1

# ─── Start Command ────────────────────────────────────────────────
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2"]
     # workers = 2x(CPU cores) + 1 for production
```

### 14.2 Docker Compose

```yaml
# docker-compose.yml

version: "3.8"

services:
  api:
    build: .
    container_name: ml-api
    ports:
      - "8000:8000"
    environment:
      - ENV=production
      - SECRET_KEY=${SECRET_KEY}
    volumes:
      - ./app/ml_models:/app/app/ml_models:ro  # Read-only model mount
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/health/live"]
      interval: 30s
      timeout: 10s
      retries: 3
    deploy:
      resources:
        limits:
          memory: 2G    # Prevent OOM
          cpus: "2.0"

  # Optional: Redis for caching predictions / rate limiting
  redis:
    image: redis:7-alpine
    container_name: ml-redis
    ports:
      - "6379:6379"
    restart: unless-stopped

  # Optional: Nginx reverse proxy
  nginx:
    image: nginx:alpine
    container_name: ml-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/conf.d/default.conf:ro
    depends_on:
      - api
```

```bash
# Build and run
docker compose up --build

# Run in background
docker compose up -d

# View logs
docker compose logs -f api

# Stop
docker compose down
```

---

## 15. 🏗️ THE PROJECT: End-to-End ML Platform

### Project: **SentimentAI API**
A production-ready API that:
- Accepts text/CSV for sentiment analysis
- Classifies as Positive / Negative / Neutral
- Returns confidence scores
- Requires API key authentication
- Supports batch processing
- Has full health monitoring

### Step 1: Train the Sentiment Model

```python
# scripts/train_sentiment_model.py
"""
Run: python scripts/train_sentiment_model.py
"""

import joblib
import json
import os
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Sample training data (replace with real dataset in production)
TRAINING_DATA = [
    ("I love this product, it's amazing!", "positive"),
    ("Absolutely fantastic experience", "positive"),
    ("Great service and fast delivery", "positive"),
    ("Best purchase I've ever made", "positive"),
    ("Really happy with this, highly recommend", "positive"),
    ("Exceeded my expectations completely", "positive"),
    ("Perfect quality, will buy again", "positive"),
    ("Outstanding value for money", "positive"),
    
    ("This is terrible, complete waste of money", "negative"),
    ("Horrible experience, never again", "negative"),
    ("Broken on arrival, very disappointed", "negative"),
    ("Worst product ever, don't buy", "negative"),
    ("Awful customer service, totally useless", "negative"),
    ("Complete garbage, falling apart already", "negative"),
    ("Defective and poor quality", "negative"),
    ("Wasted my money on this junk", "negative"),
    
    ("It's okay, nothing special", "neutral"),
    ("Average product, does what it says", "neutral"),
    ("Received the item, works as described", "neutral"),
    ("Delivery was on time, product is fine", "neutral"),
    ("Not bad but not great either", "neutral"),
    ("Mediocre at best, could be better", "neutral"),
    ("Standard quality, meets expectations", "neutral"),
    ("Nothing to complain about, nothing to praise", "neutral"),
]

# Prepare data
texts, labels = zip(*TRAINING_DATA)

# More data via augmentation (simple technique)
augmented_texts = list(texts)
augmented_labels = list(labels)
for text, label in zip(texts, labels):
    augmented_texts.append(text.lower())
    augmented_labels.append(label)
    augmented_texts.append(text.upper())
    augmented_labels.append(label)

X_train, X_test, y_train, y_test = train_test_split(
    augmented_texts, augmented_labels, test_size=0.2, random_state=42
)

# Build pipeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),     # Unigrams and bigrams
        max_features=10000,
        stop_words="english",
        min_df=1
    )),
    ("classifier", LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=42,
        multi_class="multinomial"
    ))
])

pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

accuracy = pipeline.score(X_test, y_test)
classes = pipeline.classes_.tolist()

# Save
os.makedirs("app/ml_models", exist_ok=True)
joblib.dump(pipeline, "app/ml_models/sentiment_model.pkl")

metadata = {
    "name": "sentiment_model",
    "version": "1.0.0",
    "description": "Text sentiment classifier (positive/negative/neutral)",
    "classes": classes,
    "input_type": "text",
    "accuracy": round(accuracy, 4),
    "training_samples": len(augmented_texts)
}
with open("app/ml_models/sentiment_model_meta.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\n✅ Model saved! Accuracy: {accuracy:.4f}")
print(f"   Classes: {classes}")
```

### Step 2: Sentiment Service

```python
# app/services/sentiment_service.py

import time
import joblib
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
from loguru import logger

class SentimentService:
    """Sentiment analysis inference service."""
    
    def __init__(self):
        self.model = None
        self.classes = []
        self.model_version = "unknown"
    
    def load(self, model_path: str = "app/ml_models/sentiment_model.pkl") -> bool:
        try:
            self.model = joblib.load(model_path)
            self.classes = self.model.classes_.tolist()
            self.model_version = "1.0.0"
            logger.info(f"Sentiment model loaded. Classes: {self.classes}")
            return True
        except Exception as e:
            logger.error(f"Failed to load sentiment model: {e}")
            return False
    
    def predict_single(self, text: str) -> Dict:
        """Predict sentiment for a single text."""
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        text = text.strip()
        if not text:
            raise ValueError("Text cannot be empty")
        
        start = time.time()
        
        # Predict
        proba = self.model.predict_proba([text])[0]
        predicted_idx = np.argmax(proba)
        
        inference_ms = (time.time() - start) * 1000
        
        sentiment = self.classes[predicted_idx]
        confidence = float(proba[predicted_idx])
        
        return {
            "sentiment": sentiment,
            "confidence": round(confidence, 4),
            "probabilities": {
                cls: round(float(p), 4)
                for cls, p in zip(self.classes, proba)
            },
            "model_version": self.model_version,
            "inference_time_ms": round(inference_ms, 3)
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """Efficient batch prediction."""
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        cleaned = [t.strip() for t in texts]
        
        start = time.time()
        probas = self.model.predict_proba(cleaned)
        predictions = np.argmax(probas, axis=1)
        total_ms = (time.time() - start) * 1000
        
        results = []
        for text, pred_idx, proba in zip(cleaned, predictions, probas):
            results.append({
                "text": text[:100] + "..." if len(text) > 100 else text,
                "sentiment": self.classes[pred_idx],
                "confidence": round(float(proba[pred_idx]), 4),
                "probabilities": {
                    cls: round(float(p), 4)
                    for cls, p in zip(self.classes, proba)
                }
            })
        
        logger.info(f"Batch sentiment: {len(texts)} texts in {total_ms:.1f}ms")
        return results

# Global instance
sentiment_service = SentimentService()
```

### Step 3: Sentiment Endpoints

```python
# app/api/v1/endpoints/sentiment.py

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
import io
import pandas as pd
from app.services.sentiment_service import sentiment_service
from app.core.security import get_api_key
from app.models.response_models import APIResponse

router = APIRouter(prefix="/sentiment", tags=["Sentiment Analysis"])

# ─── Request Models ───────────────────────────────────────────────
class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    
    @field_validator("text")
    @classmethod
    def clean_text(cls, v):
        return v.strip()
    
    model_config = {
        "json_schema_extra": {
            "examples": [{"text": "This product is absolutely amazing!"}]
        }
    }

class BatchTextRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=100)
    
    @field_validator("texts")
    @classmethod
    def validate_texts(cls, v):
        cleaned = [t.strip() for t in v if t.strip()]
        if not cleaned:
            raise ValueError("All texts are empty")
        return cleaned

# ─── Endpoints ────────────────────────────────────────────────────

@router.post(
    "/analyze",
    response_model=APIResponse,
    summary="Analyze text sentiment",
)
def analyze_sentiment(
    request: TextRequest,
    api_key: dict = Depends(get_api_key)   # Authentication required
):
    """
    Analyze the sentiment of a text.
    
    Returns: positive, negative, or neutral with confidence score.
    
    Requires: X-API-Key header
    """
    try:
        result = sentiment_service.predict_single(request.text)
        return APIResponse(
            success=True,
            message=f"Sentiment: {result['sentiment']}",
            data={**result, "input_text": request.text}
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/batch",
    response_model=APIResponse,
    summary="Batch sentiment analysis"
)
def batch_analyze(
    request: BatchTextRequest,
    api_key: dict = Depends(get_api_key)
):
    """Analyze sentiment for up to 100 texts in a single API call."""
    try:
        results = sentiment_service.predict_batch(request.texts)
        
        # Aggregate stats
        sentiments = [r["sentiment"] for r in results]
        stats = {
            "positive": sentiments.count("positive"),
            "negative": sentiments.count("negative"),
            "neutral": sentiments.count("neutral"),
        }
        
        return APIResponse(
            success=True,
            message=f"Analyzed {len(results)} texts",
            data={
                "results": results,
                "total": len(results),
                "summary": stats,
                "overall_sentiment": max(stats, key=stats.get)
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/analyze-csv",
    summary="Upload CSV for batch analysis"
)
async def analyze_csv(
    file: UploadFile = File(...),
    text_column: str = Form(..., description="Column name containing text"),
    api_key: dict = Depends(get_api_key)
):
    """
    Upload a CSV file, specify which column has text, get sentiment for each row.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=422, detail="File must be .csv")
    
    content = await file.read()
    
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception:
        raise HTTPException(status_code=422, detail="Could not parse CSV file")
    
    if text_column not in df.columns:
        raise HTTPException(
            status_code=422,
            detail=f"Column '{text_column}' not found. Available: {list(df.columns)}"
        )
    
    if len(df) > 5000:
        raise HTTPException(status_code=400, detail="CSV too large. Max 5000 rows.")
    
    texts = df[text_column].fillna("").astype(str).tolist()
    results = sentiment_service.predict_batch(texts)
    
    df["sentiment"] = [r["sentiment"] for r in results]
    df["confidence"] = [r["confidence"] for r in results]
    
    return {
        "success": True,
        "rows_processed": len(df),
        "sentiment_distribution": df["sentiment"].value_counts().to_dict(),
        "results": df[[text_column, "sentiment", "confidence"]].to_dict(orient="records")
    }
```

### Step 4: Wire Everything Together

```python
# app/api/v1/router.py — FINAL VERSION

from fastapi import APIRouter
from app.api.v1.endpoints import predict, health, auth, sentiment

v1_router = APIRouter(prefix="/api/v1")
v1_router.include_router(health.router)
v1_router.include_router(auth.router)
v1_router.include_router(predict.router)
v1_router.include_router(sentiment.router)
```

```python
# app/main.py — COMPLETE FINAL VERSION

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.api.v1.router import v1_router
from app.services.ml_service import model_registry
from app.services.sentiment_service import sentiment_service
from app.core.middleware import RequestLoggingMiddleware
from app.config import setup_logging

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ═══ STARTUP ════
    setup_logging()
    logger.info("🚀 Starting SentimentAI API...")
    
    # Load all models
    model_registry.load_model("iris_classifier")
    sentiment_service.load("app/ml_models/sentiment_model.pkl")
    
    logger.info("✅ All models loaded. API ready.")
    yield
    
    # ═══ SHUTDOWN ════
    logger.info("Shutting down gracefully...")


app = FastAPI(
    title="SentimentAI API",
    description="""
## 🤖 Production ML API
    
Endpoints for ML inference, sentiment analysis, and model management.

### Authentication
All prediction endpoints require an API key via `X-API-Key` header.

### Available Models
- **Iris Classifier** — Flower species classification
- **Sentiment Analyzer** — Text sentiment (positive/negative/neutral)
    """,
    version="1.0.0",
    lifespan=lifespan,
    contact={"name": "API Support", "email": "support@yourapi.com"}
)

# ─── Middleware ────────────────────────────────────────────────────
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Exception Handlers ───────────────────────────────────────────
@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    errors = [
        {"field": " → ".join(str(l) for l in e["loc"]), "message": e["msg"]}
        for e in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"success": False, "errors": errors})

@app.exception_handler(Exception)
async def generic_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"success": False, "error": "Internal Server Error"})

# ─── Routes ───────────────────────────────────────────────────────
app.include_router(v1_router)

@app.get("/", tags=["Root"])
def root():
    return {
        "name": "SentimentAI API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health/"
    }
```

### Step 5: Run Everything

```bash
# 1. Train the models
python scripts/train_sentiment_model.py
python -m app.services.train_sample_model

# 2. Start the API
uvicorn app.main:app --reload --port 8000

# 3. Test with curl
# Health check
curl http://localhost:8000/api/v1/health/

# Get API token (JWT)
curl -X POST http://localhost:8000/api/v1/auth/token \
  -d "username=alice&password=secret123"

# Analyze sentiment (with API key)
curl -X POST http://localhost:8000/api/v1/sentiment/analyze \
  -H "Content-Type: application/json" \
  -H "X-API-Key: sk-dev-1234567890abcdef" \
  -d '{"text": "This API is absolutely amazing!"}'

# Predict iris
curl -X POST http://localhost:8000/api/v1/predict/iris \
  -H "Content-Type: application/json" \
  -H "X-API-Key: sk-dev-1234567890abcdef" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2], "return_probabilities": true}'

# 4. Open interactive docs
open http://localhost:8000/docs

# 5. Run tests
pytest tests/ -v --tb=short

# 6. Build Docker image
docker build -t sentiment-api .
docker run -p 8000:8000 sentiment-api
```

---

## 🎓 What You've Learned

| Concept | Applied In |
|---------|-----------|
| FastAPI app setup & routing | `main.py`, `router.py` |
| Pydantic validation | `request_models.py`, `response_models.py` |
| Model loading & caching | `ml_service.py`, `lifespan()` |
| Sync vs Async | Prediction endpoints |
| Background tasks | Job queue pattern |
| File uploads | Image & CSV endpoints |
| API authentication | API key + JWT |
| Middleware | Logging, CORS, rate limiting |
| Error handling | Exception handlers |
| Testing | `tests/` directory |
| Dockerization | `Dockerfile`, `docker-compose.yml` |

## 🚀 Next Steps to Level Up

1. **Add a real database** — PostgreSQL + SQLAlchemy to store predictions
2. **Add Redis caching** — Cache repeated predictions with same input
3. **Add Celery** — Proper async task queue for training jobs
4. **Add Prometheus metrics** — Monitor request counts, latency, errors
5. **Deploy to cloud** — AWS EC2/ECS, GCP Cloud Run, Railway, or Render
6. **Add model versioning** — A/B test models, traffic splitting
7. **Add MLflow** — Track experiments, log models, model registry

---

*Built with ❤️ for ML engineers. Remember: a deployed model at 80% accuracy beats a perfect model never shipped.*