# 🚀 FastAPI for ML/AI Deployment — Part 2: Production Engineering
### Advanced Guide: Databases, Caching, Task Queues, Monitoring, Cloud Deployment & MLOps

> **Prerequisites:** Completed Part 1. You have a running SentimentAI API with models, auth, and Docker.
> **Goal:** Transform your working API into a production-grade, observable, scalable ML platform.

---

## Table of Contents

1. [PostgreSQL + SQLAlchemy — Persist Everything](#1-postgresql--sqlalchemy--persist-everything)
2. [Redis Caching — Make Inference Near-Instant](#2-redis-caching--make-inference-near-instant)
3. [Celery — Real Async Task Queue for Training Jobs](#3-celery--real-async-task-queue-for-training-jobs)
4. [Prometheus + Grafana — Metrics & Monitoring](#4-prometheus--grafana--metrics--monitoring)
5. [Cloud Deployment — Railway, GCP Cloud Run, AWS ECS](#5-cloud-deployment--railway-gcp-cloud-run-aws-ecs)
6. [Model Versioning — A/B Testing & Traffic Splitting](#6-model-versioning--ab-testing--traffic-splitting)
7. [MLflow — Experiment Tracking & Model Registry](#7-mlflow--experiment-tracking--model-registry)
8. [🏗️ THE PROJECT: Wiring It All Together](#8-the-project-wiring-it-all-together)

---

## 1. PostgreSQL + SQLAlchemy — Persist Everything

### Why Persist Predictions?

Without a database you can't:
- Audit what predictions were made and when
- Detect model drift (accuracy degrading over time)
- Replay failed requests
- Bill users based on usage
- Analyze which inputs come most often

### The Mental Model

```
Request → FastAPI → SQLAlchemy ORM → PostgreSQL
                        ↑
                   (Alembic migrations manage schema changes)
```

### 1.1 Install Dependencies

```bash
pip install sqlalchemy==2.0.25
pip install asyncpg==0.29.0          # Async PostgreSQL driver
pip install alembic==1.13.1          # Database migrations
pip install psycopg2-binary==2.9.9   # Sync driver (for Alembic CLI)
```

### 1.2 Database Configuration

Create `app/database.py`:

```python
# app/database.py

"""
WHY ASYNC DATABASE?
- Sync DB calls block FastAPI's event loop
- asyncpg is 3-5x faster than psycopg2 for concurrent requests
- Under load (100+ req/s), sync DB will bottleneck your entire API
"""

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import Column, String, Float, DateTime, Integer, Text, Boolean, Index
from datetime import datetime
import uuid
import os

# ─── Connection URL ───────────────────────────────────────────────
# Format: postgresql+asyncpg://user:password@host:port/dbname
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://mluser:mlpassword@localhost:5432/mldb"
)

# ─── Engine ───────────────────────────────────────────────────────
engine = create_async_engine(
    DATABASE_URL,
    echo=False,           # Set True to log all SQL (dev only)
    pool_size=10,         # Keep 10 connections open (connection pool)
    max_overflow=20,      # Allow 20 extra connections under load
    pool_pre_ping=True,   # Verify connection is alive before using it
)

# ─── Session Factory ──────────────────────────────────────────────
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False   # Keep object attributes after commit
)

# ─── Base Model ───────────────────────────────────────────────────
class Base(DeclarativeBase):
    pass

# ─── Dependency ───────────────────────────────────────────────────
async def get_db() -> AsyncSession:
    """
    FastAPI dependency that provides a database session.
    Automatically commits on success, rolls back on error.
    
    Usage in endpoint:
        @router.post("/predict")
        async def predict(db: AsyncSession = Depends(get_db)):
            ...
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
```

### 1.3 Database Models (Tables)

Create `app/db_models.py`:

```python
# app/db_models.py

from sqlalchemy import (
    Column, String, Float, DateTime, Integer,
    Text, Boolean, JSON, Index, ForeignKey
)
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
from app.database import Base

def generate_uuid():
    return str(uuid.uuid4())

# ─── Predictions Table ────────────────────────────────────────────
class Prediction(Base):
    """
    Stores every prediction made by the API.
    
    WHY STORE PREDICTIONS?
    - Audit trail: who predicted what, when
    - Feedback loop: attach ground truth labels later
    - Drift detection: compare prediction distributions over time
    - Usage analytics: most common inputs, model performance
    """
    __tablename__ = "predictions"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    
    # What was predicted
    model_name = Column(String(100), nullable=False, index=True)
    model_version = Column(String(50), nullable=False)
    input_data = Column(JSON, nullable=False)     # The raw input
    output_data = Column(JSON, nullable=False)    # The raw output
    
    # Performance
    predicted_label = Column(String(100), nullable=False)
    confidence = Column(Float, nullable=False)
    inference_time_ms = Column(Float, nullable=False)
    
    # Meta
    api_key_used = Column(String(100), nullable=True)   # Which key made this request
    client_ip = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Feedback (filled in later by monitoring/users)
    true_label = Column(String(100), nullable=True)     # Ground truth (optional)
    is_correct = Column(Boolean, nullable=True)         # Was prediction correct?
    feedback_at = Column(DateTime, nullable=True)
    
    # Indexes for common queries
    __table_args__ = (
        Index("ix_predictions_model_created", "model_name", "created_at"),
        Index("ix_predictions_label", "predicted_label"),
    )

# ─── API Keys Table ───────────────────────────────────────────────
class APIKey(Base):
    """
    Tracks API keys, their owners, and usage.
    Replaces the hardcoded VALID_API_KEYS dict from Part 1.
    """
    __tablename__ = "api_keys"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    key_hash = Column(String(256), unique=True, nullable=False)  # Never store raw keys
    key_prefix = Column(String(20), nullable=False)              # "sk-dev-..." prefix for display
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    
    tier = Column(String(50), default="free")          # free, pro, enterprise
    is_active = Column(Boolean, default=True)
    requests_count = Column(Integer, default=0)
    last_used_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)
    
    user = relationship("User", back_populates="api_keys")

# ─── Users Table ──────────────────────────────────────────────────
class User(Base):
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(100), unique=True, nullable=False)
    hashed_password = Column(String(256), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    api_keys = relationship("APIKey", back_populates="user")

# ─── Model Versions Table ─────────────────────────────────────────
class ModelVersion(Base):
    """
    Tracks deployed model versions (used in Chapter 6).
    """
    __tablename__ = "model_versions"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    model_name = Column(String(100), nullable=False, index=True)
    version = Column(String(50), nullable=False)
    file_path = Column(String(500), nullable=False)
    
    # Traffic routing
    traffic_weight = Column(Float, default=1.0)   # 0.0 to 1.0
    is_active = Column(Boolean, default=True)
    is_champion = Column(Boolean, default=False)  # The "production" model
    is_challenger = Column(Boolean, default=False) # The "test" model
    
    # Performance tracking
    total_predictions = Column(Integer, default=0)
    avg_confidence = Column(Float, nullable=True)
    accuracy = Column(Float, nullable=True)        # If feedback available
    
    metadata_ = Column("metadata", JSON, nullable=True)  # From MLflow etc.
    deployed_at = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index("ix_model_versions_name_version", "model_name", "version"),
    )
```

### 1.4 Alembic — Database Migrations

```bash
# Initialize Alembic (run once)
alembic init alembic

# This creates:
# alembic/
#   env.py       ← Configure database connection
#   versions/    ← Migration files live here
# alembic.ini    ← Alembic config
```

Edit `alembic/env.py`:

```python
# alembic/env.py — key parts to modify

from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import your models so Alembic knows about them
from app.database import Base
from app import db_models  # noqa — registers models with Base

config = context.config

# Override sqlalchemy.url from env var (sync URL for migrations)
config.set_main_option(
    "sqlalchemy.url",
    os.getenv("DATABASE_URL_SYNC", "postgresql+psycopg2://mluser:mlpassword@localhost:5432/mldb")
)

target_metadata = Base.metadata  # Tell Alembic about your tables

def run_migrations_offline():
    url = config.get_main_option("sqlalchemy.url")
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online():
    connectable = engine_from_config(
        config.config_ini_section,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

```bash
# Create first migration (auto-detects your models)
alembic revision --autogenerate -m "initial tables"

# Apply migration to database
alembic upgrade head

# Future workflow:
# 1. Change your SQLAlchemy model
# 2. alembic revision --autogenerate -m "add column X"
# 3. alembic upgrade head

# Rollback last migration
alembic downgrade -1

# See migration history
alembic history
```

### 1.5 Repository Pattern — Clean Database Access

Create `app/repositories/prediction_repo.py`:

```python
# app/repositories/prediction_repo.py

"""
WHY REPOSITORY PATTERN?
- Endpoints should NOT contain raw SQL
- Repository = single place for all DB operations on a model
- Easy to mock in tests (just mock the repo, not the DB)
- Swap databases without changing business logic
"""

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc, and_
from sqlalchemy.orm import selectinload
from typing import Optional, List, Dict
from datetime import datetime, timedelta
from app.db_models import Prediction, ModelVersion

class PredictionRepository:
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def create(
        self,
        model_name: str,
        model_version: str,
        input_data: dict,
        output_data: dict,
        predicted_label: str,
        confidence: float,
        inference_time_ms: float,
        api_key_used: Optional[str] = None,
        client_ip: Optional[str] = None,
    ) -> Prediction:
        """Save a new prediction to the database."""
        prediction = Prediction(
            model_name=model_name,
            model_version=model_version,
            input_data=input_data,
            output_data=output_data,
            predicted_label=predicted_label,
            confidence=confidence,
            inference_time_ms=inference_time_ms,
            api_key_used=api_key_used,
            client_ip=client_ip,
        )
        self.db.add(prediction)
        await self.db.flush()  # Get the ID without committing
        return prediction
    
    async def get_by_id(self, prediction_id: str) -> Optional[Prediction]:
        result = await self.db.execute(
            select(Prediction).where(Prediction.id == prediction_id)
        )
        return result.scalar_one_or_none()
    
    async def list_recent(
        self,
        model_name: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
        since: Optional[datetime] = None
    ) -> List[Prediction]:
        query = select(Prediction).order_by(desc(Prediction.created_at))
        
        if model_name:
            query = query.where(Prediction.model_name == model_name)
        if since:
            query = query.where(Prediction.created_at >= since)
        
        query = query.limit(limit).offset(offset)
        result = await self.db.execute(query)
        return result.scalars().all()
    
    async def get_stats(
        self,
        model_name: str,
        hours: int = 24
    ) -> Dict:
        """Get prediction statistics for the last N hours."""
        since = datetime.utcnow() - timedelta(hours=hours)
        
        result = await self.db.execute(
            select(
                func.count(Prediction.id).label("total"),
                func.avg(Prediction.confidence).label("avg_confidence"),
                func.avg(Prediction.inference_time_ms).label("avg_latency_ms"),
                func.min(Prediction.inference_time_ms).label("min_latency_ms"),
                func.max(Prediction.inference_time_ms).label("max_latency_ms"),
            ).where(
                and_(
                    Prediction.model_name == model_name,
                    Prediction.created_at >= since
                )
            )
        )
        row = result.one()
        
        # Label distribution
        label_counts_result = await self.db.execute(
            select(
                Prediction.predicted_label,
                func.count(Prediction.id).label("count")
            ).where(
                and_(
                    Prediction.model_name == model_name,
                    Prediction.created_at >= since
                )
            ).group_by(Prediction.predicted_label)
        )
        label_dist = {r.predicted_label: r.count for r in label_counts_result}
        
        return {
            "model_name": model_name,
            "period_hours": hours,
            "total_predictions": row.total or 0,
            "avg_confidence": round(float(row.avg_confidence or 0), 4),
            "avg_latency_ms": round(float(row.avg_latency_ms or 0), 2),
            "min_latency_ms": round(float(row.min_latency_ms or 0), 2),
            "max_latency_ms": round(float(row.max_latency_ms or 0), 2),
            "label_distribution": label_dist,
        }
    
    async def add_feedback(
        self,
        prediction_id: str,
        true_label: str
    ) -> Optional[Prediction]:
        """Add ground truth label to a past prediction."""
        prediction = await self.get_by_id(prediction_id)
        if not prediction:
            return None
        
        prediction.true_label = true_label
        prediction.is_correct = (prediction.predicted_label == true_label)
        prediction.feedback_at = datetime.utcnow()
        
        await self.db.flush()
        return prediction
```

### 1.6 Updated Endpoint with DB Logging

```python
# app/api/v1/endpoints/sentiment.py — updated predict endpoint

from fastapi import APIRouter, HTTPException, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.repositories.prediction_repo import PredictionRepository
from app.services.sentiment_service import sentiment_service
from app.core.security import get_api_key

@router.post("/analyze")
async def analyze_sentiment(
    request_body: TextRequest,
    request: Request,                         # To get client IP
    db: AsyncSession = Depends(get_db),
    api_key_info: dict = Depends(get_api_key)
):
    result = sentiment_service.predict_single(request_body.text)
    
    # Save to database — runs AFTER response data is ready
    repo = PredictionRepository(db)
    await repo.create(
        model_name="sentiment_model",
        model_version=result["model_version"],
        input_data={"text": request_body.text},
        output_data=result,
        predicted_label=result["sentiment"],
        confidence=result["confidence"],
        inference_time_ms=result["inference_time_ms"],
        api_key_used=api_key_info.get("user"),
        client_ip=request.client.host,
    )
    # DB commit happens automatically in get_db() dependency
    
    return {"success": True, "data": result}


# ─── Analytics Endpoint ───────────────────────────────────────────
@router.get("/stats/{model_name}")
async def get_model_stats(
    model_name: str,
    hours: int = 24,
    db: AsyncSession = Depends(get_db),
    api_key_info: dict = Depends(get_api_key)
):
    """Get prediction statistics for a model over the last N hours."""
    repo = PredictionRepository(db)
    stats = await repo.get_stats(model_name, hours)
    return {"success": True, "data": stats}


@router.post("/feedback/{prediction_id}")
async def submit_feedback(
    prediction_id: str,
    true_label: str,
    db: AsyncSession = Depends(get_db)
):
    """Submit the correct label for a past prediction — enables accuracy tracking."""
    repo = PredictionRepository(db)
    prediction = await repo.add_feedback(prediction_id, true_label)
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
    
    return {
        "prediction_id": prediction_id,
        "was_correct": prediction.is_correct,
        "predicted": prediction.predicted_label,
        "actual": true_label
    }
```

### 1.7 Docker Compose — Add PostgreSQL

```yaml
# docker-compose.yml — add database service

services:
  db:
    image: postgres:16-alpine
    container_name: ml-postgres
    environment:
      POSTGRES_USER: mluser
      POSTGRES_PASSWORD: mlpassword
      POSTGRES_DB: mldb
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data   # Persist data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U mluser -d mldb"]
      interval: 10s
      timeout: 5s
      retries: 5

  api:
    build: .
    depends_on:
      db:
        condition: service_healthy   # Wait for DB to be ready
    environment:
      DATABASE_URL: postgresql+asyncpg://mluser:mlpassword@db:5432/mldb
      DATABASE_URL_SYNC: postgresql+psycopg2://mluser:mlpassword@db:5432/mldb
    # ... rest of api service

volumes:
  postgres_data:
```

```bash
# Start PostgreSQL
docker compose up db -d

# Wait for it, then run migrations
alembic upgrade head

# Verify tables were created
docker exec ml-postgres psql -U mluser -d mldb -c "\dt"
```

---

## 2. Redis Caching — Make Inference Near-Instant

### Why Cache ML Predictions?

Inference is expensive. If user A asks "Is this email spam?" and 1 minute later user B sends the identical email — why run inference twice?

```
Without cache:  "I love you!" → sentiment model → 45ms → "positive"
With cache:     "I love you!" → Redis lookup → 0.3ms → "positive" (150x faster)
```

**When to cache:**
- NLP: same sentence → same result (deterministic models)
- Image classification: same image bytes → same label
- **Don't cache:** models with randomness, time-sensitive data

### 2.1 Install & Setup Redis

```bash
pip install redis==5.0.1
pip install hiredis==2.3.2   # Fast Redis parser (optional but recommended)
```

Create `app/cache.py`:

```python
# app/cache.py

"""
CACHING STRATEGY:
- Key: hash of input data (deterministic)
- Value: JSON serialized prediction result
- TTL: 1 hour (predictions don't change, but keep it reasonable)
- Namespace: model_name:version:input_hash
"""

import redis.asyncio as redis
import hashlib
import json
import os
from typing import Optional, Any
from loguru import logger
from functools import wraps

# ─── Redis Connection ─────────────────────────────────────────────
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Connection pool — reuses connections (don't create a new connection per request)
redis_pool = redis.ConnectionPool.from_url(
    REDIS_URL,
    max_connections=50,
    decode_responses=True    # Auto-decode bytes to strings
)

async def get_redis() -> redis.Redis:
    """Get a Redis client from the pool."""
    return redis.Redis(connection_pool=redis_pool)


# ─── Cache Manager ────────────────────────────────────────────────
class CacheManager:
    
    def __init__(self, default_ttl: int = 3600):
        self.default_ttl = default_ttl  # 1 hour default
    
    def _make_key(self, namespace: str, data: Any) -> str:
        """
        Create a deterministic cache key from input data.
        
        Example:
            namespace = "sentiment:v1"
            data = "I love this product"
            key = "sentiment:v1:a3f8b2c1d4e5f6a7..."
        """
        # Serialize data consistently (sorted keys for dicts)
        serialized = json.dumps(data, sort_keys=True, ensure_ascii=True)
        # MD5 is fast and good enough for cache keys (not security)
        data_hash = hashlib.md5(serialized.encode()).hexdigest()
        return f"ml_cache:{namespace}:{data_hash}"
    
    async def get(self, namespace: str, data: Any) -> Optional[dict]:
        """Try to get cached result. Returns None on miss."""
        key = self._make_key(namespace, data)
        
        try:
            client = await get_redis()
            cached = await client.get(key)
            
            if cached:
                logger.debug(f"Cache HIT: {key}")
                return json.loads(cached)
            
            logger.debug(f"Cache MISS: {key}")
            return None
        
        except redis.RedisError as e:
            # Cache failure should NEVER break the prediction
            # Degrade gracefully — just run inference instead
            logger.warning(f"Redis error on GET (degrading gracefully): {e}")
            return None
    
    async def set(
        self,
        namespace: str,
        data: Any,
        result: dict,
        ttl: Optional[int] = None
    ) -> bool:
        """Cache a result. Returns True on success."""
        key = self._make_key(namespace, data)
        ttl = ttl or self.default_ttl
        
        try:
            client = await get_redis()
            await client.setex(
                name=key,
                time=ttl,
                value=json.dumps(result)
            )
            logger.debug(f"Cache SET: {key} (TTL: {ttl}s)")
            return True
        except redis.RedisError as e:
            logger.warning(f"Redis error on SET (degrading gracefully): {e}")
            return False
    
    async def invalidate(self, namespace: str, data: Any) -> bool:
        """Remove a specific cached result."""
        key = self._make_key(namespace, data)
        try:
            client = await get_redis()
            await client.delete(key)
            return True
        except redis.RedisError:
            return False
    
    async def invalidate_namespace(self, namespace: str) -> int:
        """
        Invalidate ALL cached results for a namespace.
        Use when you deploy a new model version.
        """
        try:
            client = await get_redis()
            pattern = f"ml_cache:{namespace}:*"
            keys = await client.keys(pattern)
            if keys:
                deleted = await client.delete(*keys)
                logger.info(f"Invalidated {deleted} cache entries for {namespace}")
                return deleted
            return 0
        except redis.RedisError as e:
            logger.warning(f"Cache invalidation failed: {e}")
            return 0
    
    async def get_stats(self) -> dict:
        """Get cache statistics."""
        try:
            client = await get_redis()
            info = await client.info("stats")
            return {
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
                "hit_rate": (
                    info.get("keyspace_hits", 0) /
                    max(info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0), 1)
                ),
                "connected_clients": (await client.info("clients")).get("connected_clients", 0),
            }
        except redis.RedisError:
            return {"error": "Redis unavailable"}


# Global instance
cache = CacheManager(default_ttl=3600)
```

### 2.2 Cache Decorator — Apply to Any Function

```python
# app/cache.py — add this decorator

def cached_prediction(namespace: str, ttl: int = 3600, input_field: str = "text"):
    """
    Decorator that adds caching to any prediction function.
    
    Usage:
        @cached_prediction(namespace="sentiment:v1", ttl=3600)
        async def predict_sentiment(text: str) -> dict:
            ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Build cache key from the relevant input
            cache_input = kwargs.get(input_field) or (args[0] if args else None)
            
            # Try cache first
            cached_result = await cache.get(namespace, cache_input)
            if cached_result:
                cached_result["_cached"] = True
                cached_result["_cache_namespace"] = namespace
                return cached_result
            
            # Cache miss — run actual inference
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            
            # Store result in cache
            await cache.set(namespace, cache_input, result, ttl=ttl)
            result["_cached"] = False
            return result
        
        return wrapper
    return decorator
```

### 2.3 Integrate Cache into the Prediction Endpoint

```python
# Updated sentiment endpoint with caching

@router.post("/analyze")
async def analyze_sentiment(
    request_body: TextRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    api_key_info: dict = Depends(get_api_key)
):
    from app.cache import cache
    
    # ─── 1. Check cache ───────────────────────────────────────────
    cache_namespace = "sentiment:v1"
    cached = await cache.get(cache_namespace, request_body.text)
    
    if cached:
        # Return cached result — skip inference AND database write
        cached["_source"] = "cache"
        return {"success": True, "data": cached}
    
    # ─── 2. Run inference ─────────────────────────────────────────
    result = sentiment_service.predict_single(request_body.text)
    
    # ─── 3. Cache the result ──────────────────────────────────────
    await cache.set(cache_namespace, request_body.text, result, ttl=3600)
    
    # ─── 4. Persist to DB (background, non-blocking) ──────────────
    repo = PredictionRepository(db)
    await repo.create(
        model_name="sentiment_model",
        model_version=result["model_version"],
        input_data={"text": request_body.text},
        output_data=result,
        predicted_label=result["sentiment"],
        confidence=result["confidence"],
        inference_time_ms=result["inference_time_ms"],
        api_key_used=api_key_info.get("user"),
        client_ip=request.client.host,
    )
    
    result["_source"] = "model"
    return {"success": True, "data": result}


# Cache stats endpoint (useful for monitoring)
@router.get("/cache/stats")
async def get_cache_stats():
    from app.cache import cache
    stats = await cache.get_stats()
    return {"success": True, "data": stats}


# Invalidate cache when deploying new model
@router.post("/cache/invalidate/{namespace}")
async def invalidate_cache(
    namespace: str,
    api_key_info: dict = Depends(get_api_key)
):
    from app.cache import cache
    deleted = await cache.invalidate_namespace(namespace)
    return {"success": True, "deleted_entries": deleted}
```

### 2.4 Redis in Docker Compose

```yaml
# docker-compose.yml — add Redis

  redis:
    image: redis:7-alpine
    container_name: ml-redis
    command: redis-server --appendonly yes --maxmemory 512mb --maxmemory-policy allkeys-lru
    #         persist data     limit memory    evict least recently used when full
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  redis_data:
```

---

## 3. Celery — Real Async Task Queue for Training Jobs

### Why Celery?

FastAPI's `BackgroundTasks` is fine for quick tasks (logging, emails). But for:
- Model training (10 minutes+)
- Batch processing 100k rows
- GPU inference jobs

You need Celery:
- Tasks survive server restarts
- Multiple workers process tasks in parallel
- Retry failed tasks automatically
- Monitor tasks in real-time (Flower UI)
- Queue priority (fast predictions vs slow training)

### The Architecture

```
FastAPI → POST /train → Celery task → Redis (broker) → Celery Worker → runs training
                                                              ↓
FastAPI → GET /jobs/{id} ← Redis (results backend) ←─────────┘
```

### 3.1 Install Dependencies

```bash
pip install celery==5.3.6
pip install flower==2.0.1    # Celery monitoring UI
```

### 3.2 Celery Configuration

Create `app/worker/celery_app.py`:

```python
# app/worker/celery_app.py

from celery import Celery
import os

# ─── Celery App ───────────────────────────────────────────────────
celery_app = Celery(
    "ml_tasks",
    broker=os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/1"),
    backend=os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/2"),
    include=["app.worker.tasks"]  # Where to find task definitions
)

# ─── Configuration ────────────────────────────────────────────────
celery_app.conf.update(
    # Serialization
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    
    # Timezone
    timezone="UTC",
    enable_utc=True,
    
    # Task behavior
    task_track_started=True,      # Report when task starts (not just queued/done)
    task_acks_late=True,          # Acknowledge AFTER task completes (safer for failures)
    worker_prefetch_multiplier=1, # One task at a time per worker (better for ML)
    
    # Results
    result_expires=86400,         # Keep results for 24 hours
    
    # Routing — separate queues for different task types
    task_routes={
        "app.worker.tasks.train_model": {"queue": "training"},    # Heavy queue
        "app.worker.tasks.batch_predict": {"queue": "inference"}, # Fast queue
        "app.worker.tasks.send_email": {"queue": "notifications"},
    },
    
    # Retry policy
    task_default_retry_delay=60,  # Wait 60s before retry
    task_max_retries=3,
)
```

### 3.3 Define Tasks

Create `app/worker/tasks.py`:

```python
# app/worker/tasks.py

"""
CELERY TASK RULES:
1. Tasks must be idempotent — safe to run multiple times (due to retries)
2. Keep task inputs small — pass IDs, not large data objects
3. Update task state frequently for long tasks (progress tracking)
4. Always handle exceptions — use self.retry() for transient errors
"""

from celery import Task
from celery.utils.log import get_task_logger
from app.worker.celery_app import celery_app
import time
import json
import os

logger = get_task_logger(__name__)

# ─── Base Task Class ──────────────────────────────────────────────
class MLTask(Task):
    """Base class for all ML tasks with common error handling."""
    
    abstract = True
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        logger.error(f"Task {task_id} failed: {exc}")
        # Could send alert to Slack/PagerDuty here
    
    def on_success(self, retval, task_id, args, kwargs):
        logger.info(f"Task {task_id} completed successfully")
    
    def on_retry(self, exc, task_id, args, kwargs, einfo):
        logger.warning(f"Task {task_id} retrying due to: {exc}")


# ─── Training Task ────────────────────────────────────────────────
@celery_app.task(
    bind=True,          # self = access to task instance (for state updates)
    base=MLTask,
    name="app.worker.tasks.train_model",
    queue="training",
    max_retries=2,
    soft_time_limit=3600,  # Warn after 1 hour
    time_limit=7200,       # Kill after 2 hours
)
def train_model(self, model_config: dict, dataset_path: str) -> dict:
    """
    Long-running model training task.
    
    Args:
        model_config: dict with hyperparameters
        dataset_path: path to training data
        
    Returns:
        dict with model metrics and saved path
    """
    job_id = self.request.id
    logger.info(f"Starting training job {job_id} with config: {model_config}")
    
    try:
        # ─── Phase 1: Load Data ───────────────────────────────
        self.update_state(
            state="PROGRESS",
            meta={"phase": "loading_data", "progress": 10, "job_id": job_id}
        )
        logger.info("Loading training data...")
        time.sleep(2)  # Simulate data loading
        
        # ─── Phase 2: Preprocessing ───────────────────────────
        self.update_state(
            state="PROGRESS",
            meta={"phase": "preprocessing", "progress": 25, "job_id": job_id}
        )
        logger.info("Preprocessing features...")
        time.sleep(3)
        
        # ─── Phase 3: Training ────────────────────────────────
        n_estimators = model_config.get("n_estimators", 100)
        logger.info(f"Training with n_estimators={n_estimators}...")
        
        for epoch in range(1, 6):  # Simulate training epochs
            self.update_state(
                state="PROGRESS",
                meta={
                    "phase": "training",
                    "progress": 25 + (epoch * 10),
                    "epoch": epoch,
                    "total_epochs": 5,
                    "job_id": job_id
                }
            )
            time.sleep(2)
        
        # ─── Phase 4: Evaluation ──────────────────────────────
        self.update_state(
            state="PROGRESS",
            meta={"phase": "evaluating", "progress": 80, "job_id": job_id}
        )
        time.sleep(1)
        
        # Simulate results
        accuracy = 0.87 + (model_config.get("n_estimators", 100) / 10000)
        
        # ─── Phase 5: Save Model ──────────────────────────────
        self.update_state(
            state="PROGRESS",
            meta={"phase": "saving", "progress": 95, "job_id": job_id}
        )
        
        model_path = f"app/ml_models/trained_{job_id[:8]}.pkl"
        logger.info(f"Model saved to {model_path}")
        
        result = {
            "job_id": job_id,
            "status": "completed",
            "model_path": model_path,
            "metrics": {
                "accuracy": round(accuracy, 4),
                "f1_score": round(accuracy - 0.02, 4),
                "training_samples": 1000,
                "validation_samples": 200,
            },
            "config": model_config,
            "completed_at": time.time()
        }
        
        logger.info(f"Training complete! Accuracy: {accuracy:.4f}")
        return result
    
    except FileNotFoundError as e:
        # Don't retry if data file doesn't exist
        raise
    
    except Exception as e:
        logger.error(f"Training failed: {e}")
        # Retry for transient errors (memory issues, etc.)
        raise self.retry(exc=e, countdown=60)


# ─── Batch Prediction Task ────────────────────────────────────────
@celery_app.task(
    bind=True,
    base=MLTask,
    name="app.worker.tasks.batch_predict",
    queue="inference",
    max_retries=3,
    time_limit=300,   # 5 minutes max
)
def batch_predict_async(self, texts: list, model_name: str = "sentiment_model") -> dict:
    """
    Process large batch predictions asynchronously.
    Use this for batches > 100 items that would time out a regular HTTP request.
    """
    from app.services.sentiment_service import sentiment_service
    
    job_id = self.request.id
    total = len(texts)
    results = []
    batch_size = 50  # Process 50 at a time
    
    for i in range(0, total, batch_size):
        batch = texts[i:i + batch_size]
        
        self.update_state(
            state="PROGRESS",
            meta={
                "processed": i,
                "total": total,
                "progress": int((i / total) * 100),
                "job_id": job_id
            }
        )
        
        # Run inference on batch
        batch_results = sentiment_service.predict_batch(batch)
        results.extend(batch_results)
    
    return {
        "job_id": job_id,
        "total_processed": len(results),
        "results": results,
        "summary": {
            "positive": sum(1 for r in results if r["sentiment"] == "positive"),
            "negative": sum(1 for r in results if r["sentiment"] == "negative"),
            "neutral": sum(1 for r in results if r["sentiment"] == "neutral"),
        }
    }
```

### 3.4 API Endpoints for Tasks

```python
# app/api/v1/endpoints/jobs.py

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, Dict, Any
from celery.result import AsyncResult
from app.worker.tasks import train_model, batch_predict_async
from app.worker.celery_app import celery_app
from app.core.security import get_api_key

router = APIRouter(prefix="/jobs", tags=["Async Jobs"])


class TrainingConfig(BaseModel):
    n_estimators: int = 100
    max_depth: Optional[int] = None
    learning_rate: float = 0.1
    dataset_path: str = "data/training.csv"


class BatchPredictRequest(BaseModel):
    texts: list
    model_name: str = "sentiment_model"


@router.post("/train", status_code=202)
def start_training(
    config: TrainingConfig,
    api_key: dict = Depends(get_api_key)
):
    """
    Queue a model training job.
    Returns immediately with a job_id.
    Poll GET /jobs/{job_id} for status.
    """
    task = train_model.delay(
        model_config=config.model_dump(),
        dataset_path=config.dataset_path
    )
    
    return {
        "job_id": task.id,
        "status": "queued",
        "message": "Training job queued. Poll /jobs/{job_id} for status.",
        "poll_url": f"/api/v1/jobs/{task.id}"
    }


@router.post("/batch-predict", status_code=202)
def queue_batch_prediction(
    request: BatchPredictRequest,
    api_key: dict = Depends(get_api_key)
):
    """Queue a large batch prediction job."""
    if len(request.texts) < 100:
        raise HTTPException(
            status_code=400,
            detail="Use /sentiment/batch for small batches. This endpoint is for 100+ items."
        )
    
    task = batch_predict_async.delay(
        texts=request.texts,
        model_name=request.model_name
    )
    
    return {
        "job_id": task.id,
        "status": "queued",
        "total_items": len(request.texts),
        "poll_url": f"/api/v1/jobs/{task.id}"
    }


@router.get("/{job_id}")
def get_job_status(job_id: str):
    """
    Poll this endpoint to check job progress.
    
    Possible states:
    - PENDING: Job is in queue, not started yet
    - STARTED: Worker picked it up
    - PROGRESS: Task is running (has progress info)
    - SUCCESS: Task completed
    - FAILURE: Task failed
    - RETRY: Task is being retried
    """
    task = AsyncResult(job_id, app=celery_app)
    
    response = {
        "job_id": job_id,
        "status": task.status,
    }
    
    if task.status == "PROGRESS":
        # task.info contains what we passed to update_state(meta=...)
        response["progress"] = task.info
    
    elif task.status == "SUCCESS":
        response["result"] = task.result
    
    elif task.status == "FAILURE":
        # Don't expose raw exception to client
        response["error"] = "Task failed. Please contact support with job_id."
        response["error_type"] = type(task.result).__name__ if task.result else "Unknown"
    
    elif task.status == "PENDING":
        response["message"] = "Job is queued and waiting for a worker"
    
    return response


@router.delete("/{job_id}")
def cancel_job(job_id: str, api_key: dict = Depends(get_api_key)):
    """Cancel a queued or running job."""
    celery_app.control.revoke(job_id, terminate=True, signal="SIGTERM")
    return {"job_id": job_id, "status": "cancelled"}
```

### 3.5 Run Celery Workers

```bash
# Start the Celery worker (separate terminal)
celery -A app.worker.celery_app worker \
    --loglevel=info \
    --concurrency=4 \
    --queues=training,inference,notifications

# Start Flower monitoring UI (separate terminal)
celery -A app.worker.celery_app flower \
    --port=5555 \
    --broker=redis://localhost:6379/1

# Visit http://localhost:5555 — see all tasks, workers, queues
```

### 3.6 Docker Compose — Add Celery

```yaml
# docker-compose.yml — add worker services

  worker-inference:
    build: .
    container_name: ml-worker-inference
    command: celery -A app.worker.celery_app worker --loglevel=info --queues=inference --concurrency=4
    environment:
      CELERY_BROKER_URL: redis://redis:6379/1
      CELERY_RESULT_BACKEND: redis://redis:6379/2
      DATABASE_URL: postgresql+asyncpg://mluser:mlpassword@db:5432/mldb
    depends_on:
      - redis
      - db
    volumes:
      - ./app/ml_models:/app/app/ml_models

  worker-training:
    build: .
    container_name: ml-worker-training
    command: celery -A app.worker.celery_app worker --loglevel=info --queues=training --concurrency=1
    # Only 1 concurrent training job (uses lots of CPU/memory)
    environment:
      CELERY_BROKER_URL: redis://redis:6379/1
      CELERY_RESULT_BACKEND: redis://redis:6379/2
    depends_on:
      - redis
    volumes:
      - ./app/ml_models:/app/app/ml_models

  flower:
    build: .
    container_name: ml-flower
    command: celery -A app.worker.celery_app flower --port=5555
    ports:
      - "5555:5555"
    environment:
      CELERY_BROKER_URL: redis://redis:6379/1
    depends_on:
      - redis
```

---

## 4. Prometheus + Grafana — Metrics & Monitoring

### Why Monitoring?

Without monitoring, you're blind. Monitoring answers:
- How many requests per second is my API handling?
- What's the p99 latency of my inference?
- Which model version is performing better?
- Is there a memory leak?
- When did my error rate spike?

### The Stack

```
FastAPI → exposes /metrics → Prometheus (scrapes) → Grafana (visualizes)
```

### 4.1 Install Dependencies

```bash
pip install prometheus-fastapi-instrumentator==6.1.0
pip install prometheus-client==0.19.0
```

### 4.2 Setup Prometheus Metrics

Create `app/core/metrics.py`:

```python
# app/core/metrics.py

"""
METRICS TYPES:
- Counter: only goes up (total requests, errors)
- Gauge: goes up AND down (active connections, memory usage)
- Histogram: tracks distribution (request latency, inference time)
- Summary: like histogram but calculates quantiles on client
"""

from prometheus_client import Counter, Histogram, Gauge, Info
import time

# ─── Prediction Metrics ───────────────────────────────────────────
prediction_counter = Counter(
    name="ml_predictions_total",
    documentation="Total number of predictions made",
    labelnames=["model_name", "model_version", "predicted_label", "source"]
    # source = "model" or "cache"
)

prediction_errors = Counter(
    name="ml_prediction_errors_total",
    documentation="Total number of prediction errors",
    labelnames=["model_name", "error_type"]
)

# Histogram tracks distribution — know your p50, p95, p99 latency
inference_latency = Histogram(
    name="ml_inference_duration_seconds",
    documentation="Time spent on model inference",
    labelnames=["model_name", "model_version"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
    # Buckets define the histogram bins (in seconds)
)

# ─── Cache Metrics ────────────────────────────────────────────────
cache_hits = Counter(
    name="ml_cache_hits_total",
    documentation="Total cache hits",
    labelnames=["model_name"]
)

cache_misses = Counter(
    name="ml_cache_misses_total",
    documentation="Total cache misses",
    labelnames=["model_name"]
)

# ─── Model Health Metrics ─────────────────────────────────────────
models_loaded = Gauge(
    name="ml_models_loaded_count",
    documentation="Number of models currently loaded in memory"
)

model_confidence_avg = Gauge(
    name="ml_model_avg_confidence",
    documentation="Rolling average confidence score per model",
    labelnames=["model_name"]
)

# ─── Business Metrics ─────────────────────────────────────────────
active_training_jobs = Gauge(
    name="ml_active_training_jobs",
    documentation="Number of active training jobs"
)

# ─── Helpers ──────────────────────────────────────────────────────
class track_inference_time:
    """Context manager to time inference and record to Histogram."""
    
    def __init__(self, model_name: str, model_version: str):
        self.model_name = model_name
        self.model_version = model_version
        self.start = None
    
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start
        inference_latency.labels(
            model_name=self.model_name,
            model_version=self.model_version
        ).observe(duration)
```

### 4.3 Instrument FastAPI

```python
# app/main.py — add Prometheus instrumentation

from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

# After creating app:
Instrumentator(
    should_group_status_codes=True,     # Group 2xx, 4xx, 5xx
    should_ignore_untemplated=True,     # Ignore dynamic URL paths
    should_respect_env_var=True,        # Can disable via env var
    should_instrument_requests_inprogress=True,
    excluded_handlers=["/metrics", "/health"],  # Don't track these
    inprogress_name="http_requests_inprogress",
    inprogress_labels=True,
).instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)
# Visit /metrics to see all metrics in Prometheus format
```

### 4.4 Add Custom Metrics to Your Endpoints

```python
# Updated predict endpoint with metrics tracking

from app.core.metrics import (
    prediction_counter, prediction_errors,
    cache_hits, cache_misses, track_inference_time
)

@router.post("/analyze")
async def analyze_sentiment(request_body: TextRequest, ...):
    
    # Check cache
    cached = await cache.get("sentiment:v1", request_body.text)
    if cached:
        # Track cache hit
        cache_hits.labels(model_name="sentiment_model").inc()
        prediction_counter.labels(
            model_name="sentiment_model",
            model_version="v1",
            predicted_label=cached["sentiment"],
            source="cache"
        ).inc()
        return {"success": True, "data": cached}
    
    cache_misses.labels(model_name="sentiment_model").inc()
    
    try:
        # Track inference time with context manager
        with track_inference_time("sentiment_model", "v1"):
            result = sentiment_service.predict_single(request_body.text)
        
        # Track prediction outcome
        prediction_counter.labels(
            model_name="sentiment_model",
            model_version="v1",
            predicted_label=result["sentiment"],
            source="model"
        ).inc()
        
        await cache.set("sentiment:v1", request_body.text, result)
        return {"success": True, "data": result}
    
    except Exception as e:
        prediction_errors.labels(
            model_name="sentiment_model",
            error_type=type(e).__name__
        ).inc()
        raise
```

### 4.5 Prometheus + Grafana Docker Setup

```yaml
# docker-compose.yml — add monitoring stack

  prometheus:
    image: prom/prometheus:v2.49.0
    container_name: ml-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    command:
      - "--config.file=/etc/prometheus/prometheus.yml"
      - "--storage.tsdb.retention.time=30d"    # Keep 30 days of metrics

  grafana:
    image: grafana/grafana:10.2.0
    container_name: ml-grafana
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin123
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards:ro
    depends_on:
      - prometheus

volumes:
  prometheus_data:
  grafana_data:
```

Create `monitoring/prometheus.yml`:

```yaml
# monitoring/prometheus.yml

global:
  scrape_interval: 15s      # Scrape metrics every 15 seconds
  evaluation_interval: 15s

scrape_configs:
  - job_name: "ml-api"
    static_configs:
      - targets: ["api:8000"]    # FastAPI service name in docker network
    metrics_path: "/metrics"

  - job_name: "celery"
    static_configs:
      - targets: ["flower:5555"]
    metrics_path: "/metrics"
```

```bash
# Visit these after starting:
# http://localhost:9090 — Prometheus query UI
# http://localhost:3000 — Grafana dashboards (admin/admin123)

# Useful PromQL queries to put in Grafana:
# Request rate:        rate(http_requests_total[5m])
# Error rate:          rate(http_requests_total{status=~"5.."}[5m])
# p99 latency:         histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m]))
# Cache hit rate:      rate(ml_cache_hits_total[5m]) / (rate(ml_cache_hits_total[5m]) + rate(ml_cache_misses_total[5m]))
# Prediction volume:   rate(ml_predictions_total[1m]) * 60
```

---

## 5. Cloud Deployment — Railway, GCP Cloud Run, AWS ECS

### 5.1 Railway (Easiest — Start Here)

Railway is the fastest path to deployed. Connect GitHub, it detects Dockerfile, done.

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Initialize project
railway init

# Set environment variables
railway variables set DATABASE_URL="postgresql://..."
railway variables set REDIS_URL="redis://..."
railway variables set SECRET_KEY="your-secret-key"

# Deploy
railway up

# Your API is live at: https://your-project.railway.app
```

Create `railway.toml`:
```toml
[build]
builder = "DOCKERFILE"
dockerfilePath = "Dockerfile"

[deploy]
startCommand = "uvicorn app.main:app --host 0.0.0.0 --port $PORT"
healthcheckPath = "/api/v1/health/live"
healthcheckTimeout = 30
restartPolicyType = "ON_FAILURE"
restartPolicyMaxRetries = 3
```

### 5.2 GCP Cloud Run (Best for Variable Traffic)

Cloud Run is serverless containers — you pay per request, scales to zero.

```bash
# Prerequisites: Install gcloud CLI and authenticate
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Enable required APIs
gcloud services enable run.googleapis.com containerregistry.googleapis.com

# Build and push image
export PROJECT_ID=$(gcloud config get-value project)
export IMAGE="gcr.io/$PROJECT_ID/ml-api:latest"

docker build -t $IMAGE .
docker push $IMAGE

# Deploy to Cloud Run
gcloud run deploy ml-api \
    --image $IMAGE \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --port 8000 \
    --memory 2Gi \
    --cpu 2 \
    --min-instances 1 \       # Keep 1 warm (no cold starts)
    --max-instances 10 \      # Scale up to 10
    --concurrency 80 \        # 80 requests per container
    --set-env-vars "ENV=production,DATABASE_URL=postgresql+asyncpg://..."

# Get your URL
gcloud run services describe ml-api --region us-central1 --format "value(status.url)"
```

Create `cloudbuild.yaml` for CI/CD:

```yaml
# cloudbuild.yaml — runs on every git push

steps:
  # Run tests
  - name: "python:3.11"
    entrypoint: pip
    args: ["install", "-r", "requirements.txt"]
  
  - name: "python:3.11"
    entrypoint: pytest
    args: ["tests/", "-v", "--tb=short"]
  
  # Build and push image
  - name: "gcr.io/cloud-builders/docker"
    args: ["build", "-t", "gcr.io/$PROJECT_ID/ml-api:$COMMIT_SHA", "."]
  
  - name: "gcr.io/cloud-builders/docker"
    args: ["push", "gcr.io/$PROJECT_ID/ml-api:$COMMIT_SHA"]
  
  # Deploy to Cloud Run
  - name: "gcr.io/google.com/cloudsdktool/cloud-sdk"
    entrypoint: gcloud
    args:
      - run
      - deploy
      - ml-api
      - "--image=gcr.io/$PROJECT_ID/ml-api:$COMMIT_SHA"
      - "--region=us-central1"
      - "--platform=managed"
```

### 5.3 AWS ECS with Fargate

```bash
# Prerequisites: AWS CLI configured, ECR repository created

# Build and push to ECR
aws ecr get-login-password --region us-east-1 | \
    docker login --username AWS --password-stdin \
    123456789.dkr.ecr.us-east-1.amazonaws.com

docker build -t ml-api .
docker tag ml-api:latest 123456789.dkr.ecr.us-east-1.amazonaws.com/ml-api:latest
docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/ml-api:latest
```

Create `ecs-task-definition.json`:

```json
{
    "family": "ml-api",
    "networkMode": "awsvpc",
    "requiresCompatibilities": ["FARGATE"],
    "cpu": "1024",
    "memory": "2048",
    "executionRoleArn": "arn:aws:iam::123456789:role/ecsTaskExecutionRole",
    "containerDefinitions": [
        {
            "name": "ml-api",
            "image": "123456789.dkr.ecr.us-east-1.amazonaws.com/ml-api:latest",
            "portMappings": [
                {"containerPort": 8000, "protocol": "tcp"}
            ],
            "environment": [
                {"name": "ENV", "value": "production"}
            ],
            "secrets": [
                {
                    "name": "DATABASE_URL",
                    "valueFrom": "arn:aws:secretsmanager:us-east-1:123456789:secret:ml-api/db-url"
                }
            ],
            "logConfiguration": {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-group": "/ecs/ml-api",
                    "awslogs-region": "us-east-1",
                    "awslogs-stream-prefix": "ecs"
                }
            },
            "healthCheck": {
                "command": ["CMD-SHELL", "curl -f http://localhost:8000/api/v1/health/live || exit 1"],
                "interval": 30,
                "timeout": 10,
                "retries": 3,
                "startPeriod": 60
            }
        }
    ]
}
```

```bash
# Register task definition
aws ecs register-task-definition --cli-input-json file://ecs-task-definition.json

# Create service
aws ecs create-service \
    --cluster ml-cluster \
    --service-name ml-api \
    --task-definition ml-api \
    --desired-count 2 \
    --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=ENABLED}"
```

### 5.4 Environment Variables — The Right Way

```python
# app/config.py — Centralized configuration

from pydantic_settings import BaseSettings, SettingsConfigDict  # pip install pydantic-settings
from typing import Optional

class Settings(BaseSettings):
    """
    All configuration from environment variables.
    Pydantic validates types automatically.
    Use .env file locally, real env vars in production.
    """
    
    # App
    app_name: str = "ML API"
    env: str = "development"
    debug: bool = False
    secret_key: str = "change-me-in-production"
    
    # Database
    database_url: str = "postgresql+asyncpg://mluser:mlpassword@localhost:5432/mldb"
    database_url_sync: str = "postgresql+psycopg2://mluser:mlpassword@localhost:5432/mldb"
    
    # Redis
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl: int = 3600
    
    # Celery
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"
    
    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"
    
    # Models
    models_dir: str = "app/ml_models"
    models_to_load: list = ["sentiment_model", "iris_classifier"]
    
    # Security
    access_token_expire_minutes: int = 30
    
    model_config = SettingsConfigDict(
        env_file=".env",          # Load from .env file in development
        env_file_encoding="utf-8",
        case_sensitive=False
    )

# Single instance used across the app
settings = Settings()
```

Create `.env` (add to `.gitignore`!):

```bash
# .env — NEVER commit this file

ENV=development
DEBUG=true
SECRET_KEY=dev-secret-key-change-in-production
DATABASE_URL=postgresql+asyncpg://mluser:mlpassword@localhost:5432/mldb
REDIS_URL=redis://localhost:6379/0
MLFLOW_TRACKING_URI=http://localhost:5000
```

---

## 6. Model Versioning — A/B Testing & Traffic Splitting

### Why Version Models?

```
v1 model: 82% accuracy  ← "champion" (serves 90% of traffic)
v2 model: 89% accuracy  ← "challenger" (serves 10% of traffic — verifying in prod)
```

If v2 proves better in production, promote it to champion. Zero downtime.

### 6.1 Model Registry Service

Create `app/services/model_registry_service.py`:

```python
# app/services/model_registry_service.py

"""
MODEL VERSIONING CONCEPTS:
- Champion: Current production model (proven, trusted)
- Challenger: New model being tested (small traffic slice)
- Shadow: Gets requests but responses are NOT returned to user (just logged)
- Canary: Gets small % of traffic with full exposure
"""

import random
import time
import joblib
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from loguru import logger
from pathlib import Path

@dataclass
class ModelEntry:
    name: str
    version: str
    model: object              # The loaded sklearn/torch model
    traffic_weight: float      # 0.0 to 1.0
    is_champion: bool
    is_challenger: bool
    total_predictions: int = 0
    total_confidence: float = 0.0
    
    @property
    def avg_confidence(self) -> float:
        if self.total_predictions == 0:
            return 0.0
        return self.total_confidence / self.total_predictions


class VersionedModelRegistry:
    """
    Manages multiple versions of the same model with traffic routing.
    """
    
    def __init__(self):
        # Structure: {model_name: [ModelEntry, ...]}
        self._versions: Dict[str, List[ModelEntry]] = {}
    
    def register(
        self,
        name: str,
        version: str,
        model_path: str,
        traffic_weight: float = 1.0,
        is_champion: bool = False,
        is_challenger: bool = False,
    ) -> bool:
        """Load and register a model version."""
        try:
            model = joblib.load(model_path)
            
            entry = ModelEntry(
                name=name,
                version=version,
                model=model,
                traffic_weight=traffic_weight,
                is_champion=is_champion,
                is_challenger=is_challenger,
            )
            
            if name not in self._versions:
                self._versions[name] = []
            
            # Remove existing entry with same version
            self._versions[name] = [
                v for v in self._versions[name] if v.version != version
            ]
            self._versions[name].append(entry)
            
            logger.info(f"Registered model '{name}' v{version} "
                       f"(weight={traffic_weight}, champion={is_champion})")
            return True
        
        except Exception as e:
            logger.error(f"Failed to register model '{name}' v{version}: {e}")
            return False
    
    def route_request(self, model_name: str) -> Optional[ModelEntry]:
        """
        Select which model version handles this request based on traffic weights.
        
        Example:
            v1 weight=0.9, v2 weight=0.1
            → 90% of calls get v1, 10% get v2
        """
        versions = self._versions.get(model_name, [])
        active = [v for v in versions if v.traffic_weight > 0]
        
        if not active:
            return None
        
        if len(active) == 1:
            return active[0]
        
        # Weighted random selection
        total_weight = sum(v.traffic_weight for v in active)
        weights = [v.traffic_weight / total_weight for v in active]
        
        selected = random.choices(active, weights=weights, k=1)[0]
        return selected
    
    def get_champion(self, model_name: str) -> Optional[ModelEntry]:
        """Always get the champion model (bypass A/B testing)."""
        versions = self._versions.get(model_name, [])
        champions = [v for v in versions if v.is_champion]
        return champions[0] if champions else None
    
    def update_traffic(self, model_name: str, version: str, new_weight: float):
        """Adjust traffic weight dynamically (no restart needed)."""
        versions = self._versions.get(model_name, [])
        for entry in versions:
            if entry.version == version:
                old_weight = entry.traffic_weight
                entry.traffic_weight = new_weight
                logger.info(f"Traffic updated: {model_name} v{version}: "
                           f"{old_weight:.0%} → {new_weight:.0%}")
                return True
        return False
    
    def promote_to_champion(self, model_name: str, version: str):
        """Promote a challenger to champion. Demote the old champion."""
        versions = self._versions.get(model_name, [])
        for entry in versions:
            if entry.version == version:
                # Promote this one
                entry.is_champion = True
                entry.is_challenger = False
                entry.traffic_weight = 1.0
            else:
                # Demote old champion
                entry.is_champion = False
                entry.traffic_weight = 0.0  # Stop sending traffic
        
        logger.info(f"Promoted '{model_name}' v{version} to champion")
    
    def get_status(self, model_name: str) -> List[Dict]:
        versions = self._versions.get(model_name, [])
        return [
            {
                "version": v.version,
                "traffic_weight": v.traffic_weight,
                "is_champion": v.is_champion,
                "is_challenger": v.is_challenger,
                "total_predictions": v.total_predictions,
                "avg_confidence": round(v.avg_confidence, 4),
            }
            for v in versions
        ]
    
    def record_prediction(self, model_name: str, version: str, confidence: float):
        """Track per-version statistics."""
        versions = self._versions.get(model_name, [])
        for entry in versions:
            if entry.version == version:
                entry.total_predictions += 1
                entry.total_confidence += confidence
                break


versioned_registry = VersionedModelRegistry()
```

### 6.2 A/B Testing Endpoint

```python
# app/api/v1/endpoints/model_versions.py

from fastapi import APIRouter, HTTPException, Depends
from app.services.model_registry_service import versioned_registry
from app.core.security import get_api_key

router = APIRouter(prefix="/models", tags=["Model Versioning"])


@router.get("/{model_name}/versions")
def get_model_versions(model_name: str):
    """Get all versions and current traffic split for a model."""
    status = versioned_registry.get_status(model_name)
    if not status:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    return {"model_name": model_name, "versions": status}


@router.post("/{model_name}/versions/{version}/load")
def load_model_version(
    model_name: str,
    version: str,
    model_path: str,
    traffic_weight: float = 0.0,
    is_challenger: bool = True,
    api_key: dict = Depends(get_api_key)
):
    """
    Load a new model version.
    By default: traffic_weight=0.0 (shadow mode, no live traffic yet).
    """
    success = versioned_registry.register(
        name=model_name,
        version=version,
        model_path=model_path,
        traffic_weight=traffic_weight,
        is_challenger=is_challenger,
    )
    if not success:
        raise HTTPException(status_code=500, detail="Failed to load model")
    return {"message": f"Model {model_name} v{version} loaded with {traffic_weight:.0%} traffic"}


@router.patch("/{model_name}/traffic")
def update_traffic_split(
    model_name: str,
    version: str,
    traffic_weight: float,
    api_key: dict = Depends(get_api_key)
):
    """
    Adjust traffic split without restart.
    
    Example A/B test:
        PATCH /models/sentiment_model/traffic?version=v1&traffic_weight=0.9
        PATCH /models/sentiment_model/traffic?version=v2&traffic_weight=0.1
    """
    success = versioned_registry.update_traffic(model_name, version, traffic_weight)
    if not success:
        raise HTTPException(status_code=404, detail="Model version not found")
    return {"message": f"Traffic updated: {model_name} v{version} → {traffic_weight:.0%}"}


@router.post("/{model_name}/versions/{version}/promote")
def promote_to_champion(
    model_name: str,
    version: str,
    api_key: dict = Depends(get_api_key)
):
    """
    Graduate a challenger to champion — send it 100% of traffic.
    Demotes the old champion.
    """
    versioned_registry.promote_to_champion(model_name, version)
    return {
        "message": f"'{model_name}' v{version} is now the champion",
        "current_state": versioned_registry.get_status(model_name)
    }
```

### 6.3 Using Versioned Models in Prediction

```python
# Updated prediction — automatically routes to correct version

@router.post("/analyze")
async def analyze_with_versioning(request_body: TextRequest, ...):
    
    # Route to the appropriate version based on traffic weights
    model_entry = versioned_registry.route_request("sentiment_model")
    
    if not model_entry:
        raise HTTPException(status_code=503, detail="No model versions available")
    
    # Run prediction using selected version
    start = time.time()
    proba = model_entry.model.predict_proba([request_body.text])[0]
    predicted_idx = proba.argmax()
    classes = model_entry.model.classes_
    
    result = {
        "sentiment": classes[predicted_idx],
        "confidence": float(proba[predicted_idx]),
        "model_version": model_entry.version,  # Tell client which version served them
        "inference_time_ms": (time.time() - start) * 1000
    }
    
    # Track per-version stats
    versioned_registry.record_prediction(
        "sentiment_model",
        model_entry.version,
        result["confidence"]
    )
    
    return {"success": True, "data": result}
```

---

## 7. MLflow — Experiment Tracking & Model Registry

### Why MLflow?

Without experiment tracking, you lose:
- Which hyperparameters gave 89% accuracy last week?
- Which training dataset was used for model v3?
- How do my 20 experiments compare?
- Where is the model file for experiment 12?

MLflow is the answer to all of this.

### 7.1 Install & Start MLflow Server

```bash
pip install mlflow==2.10.0
pip install boto3  # If storing artifacts in S3

# Start MLflow tracking server
mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri postgresql://mluser:mlpassword@localhost:5432/mlflow_db \
    --default-artifact-root ./mlflow-artifacts

# Visit http://localhost:5000 — MLflow UI
```

### 7.2 Instrument Your Training Script

```python
# scripts/train_with_mlflow.py

"""
MLFLOW CONCEPTS:
- Experiment: A project (e.g., "sentiment_model")
- Run: One training attempt with specific parameters
- Params: Hyperparameters you set (n_estimators=100)
- Metrics: Performance numbers (accuracy=0.89)
- Artifacts: Files produced (model.pkl, confusion_matrix.png)
- Tags: Free-form labels (team="ml", dataset_version="v2")
"""

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import json
import os

# ─── Configuration ────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "sentiment_analysis"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# ─── Training Data ────────────────────────────────────────────────
TEXTS = [
    "I love this product, it's amazing!", "Absolutely fantastic experience",
    "Great service and fast delivery", "Best purchase I've ever made",
    "Really happy with this, highly recommend", "Exceeded my expectations",
    "This is terrible, complete waste of money", "Horrible experience, never again",
    "Broken on arrival, very disappointed", "Worst product ever, don't buy",
    "Awful customer service, totally useless", "Complete garbage",
    "It's okay, nothing special", "Average product, does what it says",
    "Received the item, works as described", "Not bad but not great either",
    "Mediocre at best, could be better", "Standard quality, meets expectations",
]
LABELS = (
    ["positive"] * 6 + ["negative"] * 6 + ["neutral"] * 6
)


def train_experiment(
    n_estimators: int = 100,       # Not used in LR but kept for API consistency
    C: float = 1.0,                # LR regularization
    max_features: int = 5000,
    ngram_max: int = 2,
    run_name: str = None
):
    """Train one experiment and log everything to MLflow."""
    
    with mlflow.start_run(run_name=run_name or f"C={C}_features={max_features}"):
        
        # ─── Log Parameters ───────────────────────────────────
        mlflow.log_params({
            "C": C,
            "max_features": max_features,
            "ngram_range": f"(1, {ngram_max})",
            "model_type": "LogisticRegression",
            "vectorizer": "TfidfVectorizer",
        })
        
        mlflow.set_tags({
            "dataset_version": "v1",
            "engineer": "ml-team",
            "task": "sentiment_classification",
        })
        
        # ─── Prepare Data ─────────────────────────────────────
        X_train, X_test, y_train, y_test = train_test_split(
            TEXTS, LABELS, test_size=0.2, random_state=42, stratify=LABELS
        )
        
        # ─── Build & Train Pipeline ───────────────────────────
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, ngram_max),
                max_features=max_features,
                stop_words="english"
            )),
            ("classifier", LogisticRegression(
                C=C, max_iter=1000, random_state=42
            ))
        ])
        
        pipeline.fit(X_train, y_train)
        
        # ─── Evaluate ─────────────────────────────────────────
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        f1_weighted = f1_score(y_test, y_pred, average="weighted")
        
        # Cross-validation for more robust estimate
        cv_scores = cross_val_score(pipeline, TEXTS, LABELS, cv=3, scoring="accuracy")
        
        # ─── Log Metrics ──────────────────────────────────────
        mlflow.log_metrics({
            "accuracy": round(accuracy, 4),
            "f1_macro": round(f1_macro, 4),
            "f1_weighted": round(f1_weighted, 4),
            "cv_accuracy_mean": round(cv_scores.mean(), 4),
            "cv_accuracy_std": round(cv_scores.std(), 4),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        })
        
        # ─── Log Artifacts ────────────────────────────────────
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        with open("/tmp/classification_report.json", "w") as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact("/tmp/classification_report.json")
        
        # Confusion matrix plot
        fig, ax = plt.subplots(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred, labels=pipeline.classes_)
        disp = ConfusionMatrixDisplay(cm, display_labels=pipeline.classes_)
        disp.plot(ax=ax, cmap="Blues")
        ax.set_title(f"Confusion Matrix (Accuracy: {accuracy:.2%})")
        plt.tight_layout()
        plt.savefig("/tmp/confusion_matrix.png", dpi=100, bbox_inches="tight")
        mlflow.log_artifact("/tmp/confusion_matrix.png")
        plt.close()
        
        # ─── Log Model ────────────────────────────────────────
        # Input/output schema for the model
        from mlflow.models.signature import infer_signature
        signature = infer_signature(
            model_input=X_train,
            model_output=pipeline.predict(X_train)
        )
        
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            signature=signature,
            input_example=["This product is amazing!"],
            registered_model_name="sentiment_classifier",  # Registers in Model Registry
        )
        
        run_id = mlflow.active_run().info.run_id
        print(f"\nRun ID: {run_id}")
        print(f"Accuracy: {accuracy:.4f} | F1: {f1_macro:.4f} | CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        return run_id, accuracy


# ─── Run Hyperparameter Search ────────────────────────────────────
if __name__ == "__main__":
    print("🔬 Running hyperparameter experiments...")
    
    experiments = [
        {"C": 0.1, "max_features": 5000, "run_name": "low_C"},
        {"C": 1.0, "max_features": 5000, "run_name": "default"},
        {"C": 10.0, "max_features": 5000, "run_name": "high_C"},
        {"C": 1.0, "max_features": 10000, "run_name": "more_features"},
        {"C": 1.0, "max_features": 5000, "ngram_max": 1, "run_name": "unigrams_only"},
    ]
    
    results = []
    for exp in experiments:
        run_id, accuracy = train_experiment(**exp)
        results.append({"run_id": run_id, "accuracy": accuracy, **exp})
    
    # Find best experiment
    best = max(results, key=lambda x: x["accuracy"])
    print(f"\n🏆 Best run: {best['run_name']} (accuracy={best['accuracy']:.4f})")
    print(f"   Run ID: {best['run_id']}")
    print(f"\nView results at: http://localhost:5000")
```

### 7.3 Load Models from MLflow Registry

```python
# app/services/mlflow_service.py

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from loguru import logger
import os

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

client = MlflowClient()

def load_production_model(model_name: str):
    """
    Load the model currently tagged as 'Production' in MLflow Registry.
    
    MLflow Model Registry stages:
    - None: Just registered
    - Staging: Being tested
    - Production: Live, serving traffic
    - Archived: Old version, kept for audit
    """
    try:
        model_uri = f"models:/{model_name}/Production"
        model = mlflow.sklearn.load_model(model_uri)
        
        # Get version info
        versions = client.get_latest_versions(model_name, stages=["Production"])
        version_info = versions[0] if versions else None
        
        logger.info(
            f"Loaded production model '{model_name}' "
            f"v{version_info.version if version_info else 'unknown'} "
            f"from MLflow"
        )
        return model, version_info
    
    except Exception as e:
        logger.error(f"Failed to load '{model_name}' from MLflow: {e}")
        return None, None


def load_model_by_version(model_name: str, version: int):
    """Load a specific version number from MLflow."""
    model_uri = f"models:/{model_name}/{version}"
    return mlflow.sklearn.load_model(model_uri)


def get_model_metrics(model_name: str, version: int) -> dict:
    """Get metrics logged for a specific model version."""
    versions = client.search_model_versions(f"name='{model_name}'")
    for v in versions:
        if v.version == str(version):
            run = client.get_run(v.run_id)
            return {
                "version": version,
                "run_id": v.run_id,
                "metrics": run.data.metrics,
                "params": run.data.params,
                "tags": run.data.tags,
                "status": v.current_stage,
            }
    return {}


def promote_model_to_production(model_name: str, version: int):
    """
    Promote a model version to Production stage.
    Also archives the old production version.
    """
    # Archive current production
    current_prod = client.get_latest_versions(model_name, stages=["Production"])
    for old_version in current_prod:
        client.transition_model_version_stage(
            name=model_name,
            version=old_version.version,
            stage="Archived"
        )
        logger.info(f"Archived {model_name} v{old_version.version}")
    
    # Promote new version
    client.transition_model_version_stage(
        name=model_name,
        version=str(version),
        stage="Production"
    )
    logger.info(f"Promoted {model_name} v{version} to Production")


# MLflow endpoints
def list_registered_models() -> list:
    return [
        {
            "name": rm.name,
            "latest_versions": [
                {
                    "version": v.version,
                    "stage": v.current_stage,
                    "run_id": v.run_id,
                }
                for v in client.get_latest_versions(rm.name)
            ]
        }
        for rm in client.search_registered_models()
    ]
```

### 7.4 MLflow API Endpoints

```python
# app/api/v1/endpoints/mlflow_endpoints.py

from fastapi import APIRouter, HTTPException, Depends
from app.services.mlflow_service import (
    get_model_metrics, promote_model_to_production,
    list_registered_models, load_production_model
)
from app.core.security import get_api_key

router = APIRouter(prefix="/mlflow", tags=["MLflow"])

@router.get("/models")
def list_models(api_key: dict = Depends(get_api_key)):
    """List all registered models and their versions."""
    return {"models": list_registered_models()}

@router.get("/models/{model_name}/versions/{version}/metrics")
def get_metrics(model_name: str, version: int):
    """Get training metrics for a specific model version."""
    metrics = get_model_metrics(model_name, version)
    if not metrics:
        raise HTTPException(status_code=404, detail="Model version not found")
    return metrics

@router.post("/models/{model_name}/versions/{version}/promote")
def promote_model(
    model_name: str,
    version: int,
    api_key: dict = Depends(get_api_key)
):
    """
    Promote a model version to Production.
    This is the 'deploy a new model' operation.
    """
    promote_model_to_production(model_name, version)
    
    # Reload the model in the running API
    model, version_info = load_production_model(model_name)
    if model is None:
        raise HTTPException(status_code=500, detail="Promoted but failed to reload")
    
    return {
        "message": f"'{model_name}' v{version} is now in Production",
        "reloaded": True
    }

@router.post("/models/{model_name}/reload")
def reload_production_model(
    model_name: str,
    api_key: dict = Depends(get_api_key)
):
    """Hot-reload the production model without restarting the server."""
    from app.services.sentiment_service import sentiment_service
    
    model, version_info = load_production_model(model_name)
    if model is None:
        raise HTTPException(status_code=500, detail="Failed to load model from MLflow")
    
    # Update the running service
    sentiment_service.model = model
    sentiment_service.model_version = version_info.version if version_info else "unknown"
    
    return {
        "message": f"Reloaded '{model_name}'",
        "version": sentiment_service.model_version
    }
```

### 7.5 MLflow in Docker Compose

```yaml
# docker-compose.yml — add MLflow

  mlflow:
    image: python:3.11-slim
    container_name: ml-mlflow
    command: >
      sh -c "pip install mlflow psycopg2-binary &&
             mlflow server
             --host 0.0.0.0
             --port 5000
             --backend-store-uri postgresql://mluser:mlpassword@db:5432/mlflow_db
             --default-artifact-root /mlflow/artifacts"
    ports:
      - "5000:5000"
    volumes:
      - mlflow_artifacts:/mlflow/artifacts
    depends_on:
      db:
        condition: service_healthy

volumes:
  mlflow_artifacts:
```

---

## 8. 🏗️ THE PROJECT: Wiring It All Together

### Final Architecture

```
                    ┌─────────────────────────────────────┐
                    │          NGINX (reverse proxy)       │
                    │   /api → FastAPI  :8000              │
                    │   /flower → Celery UI :5555          │
                    │   /grafana → Grafana :3000            │
                    └──────────────┬──────────────────────┘
                                   │
              ┌────────────────────▼────────────────────┐
              │             FastAPI (2 instances)         │
              │  • Pydantic validation                    │
              │  • API key / JWT auth                     │
              │  • Prometheus metrics /metrics            │
              │  • Versioned model routing                │
              └──┬───────────────┬──────────────────┬───┘
                 │               │                  │
         ┌───────▼──────┐ ┌─────▼──────┐ ┌────────▼──────┐
         │  PostgreSQL  │ │   Redis     │ │    Celery      │
         │  Predictions │ │   Cache     │ │    Workers     │
         │  Users       │ │   Sessions  │ │    Training    │
         │  API Keys    │ │             │ │    Batch jobs  │
         └──────────────┘ └────────────┘ └───────────────┘
                                                    │
                                          ┌─────────▼──────┐
                                          │    MLflow       │
                                          │  Experiments    │
                                          │  Model Registry │
                                          └────────────────┘
```

### Final `docker-compose.yml`

```yaml
# docker-compose.yml — COMPLETE PRODUCTION SETUP

version: "3.8"

services:
  # ─── Database ────────────────────────────────────────────────
  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: mluser
      POSTGRES_PASSWORD: mlpassword
      POSTGRES_MULTIPLE_DATABASES: mldb,mlflow_db
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init-multi-db.sh:/docker-entrypoint-initdb.d/init.sh
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U mluser"]
      interval: 10s
      retries: 5

  # ─── Cache & Broker ──────────────────────────────────────────
  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes --maxmemory 512mb --maxmemory-policy allkeys-lru
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      retries: 5

  # ─── API ─────────────────────────────────────────────────────
  api:
    build: .
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2
    ports:
      - "8000:8000"
    environment:
      ENV: production
      DATABASE_URL: postgresql+asyncpg://mluser:mlpassword@db:5432/mldb
      DATABASE_URL_SYNC: postgresql+psycopg2://mluser:mlpassword@db:5432/mldb
      REDIS_URL: redis://redis:6379/0
      CELERY_BROKER_URL: redis://redis:6379/1
      CELERY_RESULT_BACKEND: redis://redis:6379/2
      MLFLOW_TRACKING_URI: http://mlflow:5000
      SECRET_KEY: ${SECRET_KEY}
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_healthy
    volumes:
      - ./app/ml_models:/app/app/ml_models
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 2G

  # ─── Celery Workers ──────────────────────────────────────────
  worker-inference:
    build: .
    command: celery -A app.worker.celery_app worker -Q inference --concurrency=4 --loglevel=info
    environment:
      CELERY_BROKER_URL: redis://redis:6379/1
      CELERY_RESULT_BACKEND: redis://redis:6379/2
      DATABASE_URL: postgresql+asyncpg://mluser:mlpassword@db:5432/mldb
    depends_on: [redis, db]
    volumes:
      - ./app/ml_models:/app/app/ml_models

  worker-training:
    build: .
    command: celery -A app.worker.celery_app worker -Q training --concurrency=1 --loglevel=info
    environment:
      CELERY_BROKER_URL: redis://redis:6379/1
      CELERY_RESULT_BACKEND: redis://redis:6379/2
      MLFLOW_TRACKING_URI: http://mlflow:5000
      DATABASE_URL: postgresql+asyncpg://mluser:mlpassword@db:5432/mldb
    depends_on: [redis, db, mlflow]
    volumes:
      - ./app/ml_models:/app/app/ml_models

  # ─── Monitoring ──────────────────────────────────────────────
  flower:
    build: .
    command: celery -A app.worker.celery_app flower --port=5555
    ports:
      - "5555:5555"
    environment:
      CELERY_BROKER_URL: redis://redis:6379/1
    depends_on: [redis]

  prometheus:
    image: prom/prometheus:v2.49.0
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus

  grafana:
    image: grafana/grafana:10.2.0
    ports:
      - "3001:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin123
    volumes:
      - grafana_data:/var/lib/grafana
    depends_on: [prometheus]

  # ─── MLflow ───────────────────────────────────────────────────
  mlflow:
    image: python:3.11-slim
    command: >
      sh -c "pip install mlflow psycopg2-binary &&
             mlflow server --host 0.0.0.0 --port 5000
             --backend-store-uri postgresql://mluser:mlpassword@db:5432/mlflow_db
             --default-artifact-root /mlflow/artifacts"
    ports:
      - "5000:5000"
    volumes:
      - mlflow_artifacts:/mlflow/artifacts
    depends_on:
      db:
        condition: service_healthy

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:
  mlflow_artifacts:
```

### Startup Runbook

```bash
# ─── 1. Environment ───────────────────────────────────────────
cp .env.example .env
# Edit .env with your values
export SECRET_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")

# ─── 2. Start Infrastructure ──────────────────────────────────
docker compose up db redis mlflow -d
sleep 10  # Wait for DB to initialize

# ─── 3. Run Migrations ────────────────────────────────────────
alembic upgrade head

# ─── 4. Train Models ──────────────────────────────────────────
python scripts/train_with_mlflow.py
python -m app.services.train_sample_model

# Promote best model in MLflow UI or CLI:
# http://localhost:5000 → sentiment_classifier → Promote v1 to Production

# ─── 5. Start Everything ──────────────────────────────────────
docker compose up -d

# ─── 6. Verify ────────────────────────────────────────────────
docker compose ps             # All services should be "healthy"
curl http://localhost:8000/api/v1/health/

# ─── 7. Access Dashboards ─────────────────────────────────────
# API Docs:      http://localhost:8000/docs
# MLflow:        http://localhost:5000
# Celery Flower: http://localhost:5555
# Grafana:       http://localhost:3001  (admin/admin123)
# Prometheus:    http://localhost:9090
```

### Complete End-to-End Test

```bash
# 1. Get API token
TOKEN=$(curl -s -X POST http://localhost:8000/api/v1/auth/token \
  -d "username=alice&password=secret123" | python -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

# 2. Analyze sentiment (real-time, cached after first call)
curl -X POST http://localhost:8000/api/v1/sentiment/analyze \
  -H "Content-Type: application/json" \
  -H "X-API-Key: sk-dev-1234567890abcdef" \
  -d '{"text": "This ML platform is absolutely incredible!"}'

# 3. Queue a large batch job
curl -X POST http://localhost:8000/api/v1/jobs/batch-predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: sk-dev-1234567890abcdef" \
  -d '{"texts": ["Great!", "Terrible!", "Okay.", ...], "model_name": "sentiment_model"}'

# 4. Poll for results
curl http://localhost:8000/api/v1/jobs/{job_id}

# 5. Check model stats (uses PostgreSQL)
curl http://localhost:8000/api/v1/sentiment/stats/sentiment_model?hours=1 \
  -H "X-API-Key: sk-dev-1234567890abcdef"

# 6. Check cache performance
curl http://localhost:8000/api/v1/sentiment/cache/stats

# 7. Check metrics (Prometheus format)
curl http://localhost:8000/metrics

# 8. Start a training job via Celery
curl -X POST http://localhost:8000/api/v1/jobs/train \
  -H "Content-Type: application/json" \
  -H "X-API-Key: sk-dev-1234567890abcdef" \
  -d '{"C": 5.0, "max_features": 10000, "dataset_path": "data/training.csv"}'

# 9. Check active A/B test
curl http://localhost:8000/api/v1/models/sentiment_model/versions

# 10. Promote a model version
curl -X POST http://localhost:8000/api/v1/mlflow/models/sentiment_classifier/versions/2/promote \
  -H "X-API-Key: sk-dev-1234567890abcdef"
```

---

## 🎓 What You've Built: The Complete Picture

| Layer | Technology | Purpose |
|-------|-----------|---------|
| API Framework | FastAPI | HTTP routing, validation, docs |
| Data Validation | Pydantic v2 | Request/response schemas |
| ASGI Server | Uvicorn | High-performance async server |
| Database | PostgreSQL + SQLAlchemy async | Persist predictions, users, keys |
| Migrations | Alembic | Schema versioning |
| Caching | Redis | Skip redundant inference |
| Task Queue | Celery + Redis | Async jobs, training |
| Queue Monitor | Flower | Real-time Celery visibility |
| Metrics | Prometheus + Grafana | Observability, alerting |
| Experiment Tracking | MLflow | Hyperparameter search, model lineage |
| Model Registry | MLflow Registry | Staging → Production promotions |
| Model Versioning | Custom Registry | A/B testing, traffic splitting |
| Containerization | Docker + Compose | Reproducible deployments |
| Cloud Deploy | Railway / Cloud Run / ECS | Public internet access |

---

## 🚀 What to Build Next

**Real-World Projects:**
1. **Fraud Detection API** — PostgreSQL stores transactions, Redis flags suspicious IPs, Celery runs nightly retraining
2. **Document Classification** — PDF upload, async processing, MLflow tracks BERT fine-tuning runs
3. **Recommendation Engine** — User embeddings in PostgreSQL, Redis caches recommendations, A/B tests different algorithms
4. **Time Series Forecasting** — Prometheus metrics AS training data, Grafana dashboard shows forecast vs actual

**Production Hardening:**
- Add `Alembic` auto-migration on startup
- Set up `PagerDuty` alerts from Grafana when error rate > 1%
- Add `OpenTelemetry` distributed tracing
- Implement `circuit breaker` pattern (if model fails, serve cached fallback)
- Add `feature store` (Feast or Tecton) for ML feature reuse

---

*Part 1 gave you a working API. Part 2 gave you a production platform. The difference between these is the difference between a demo and a business.*