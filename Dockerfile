# ─── Base ─────────────────────────────────────────────────────────
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ENV=production \
    MODEL_PATH=models/model.pt

WORKDIR /app

# System deps for Pillow / torch runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# ─── Python deps ──────────────────────────────────────────────────
# Install the CPU-only torch/torchvision wheels first (smaller image),
# then the rest of the requirements.
COPY requirements.txt pyproject.toml README.md ./
RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 \
        --index-url https://download.pytorch.org/whl/cpu \
    && grep -vE '^(torch|torchvision)==' requirements.txt > /tmp/reqs.txt \
    && pip install --no-cache-dir -r /tmp/reqs.txt

# ─── App code ─────────────────────────────────────────────────────
COPY app/ ./app/
COPY pulmoscan/ ./pulmoscan/
COPY config/ ./config/
COPY params.yaml ./
RUN mkdir -p logs models && pip install --no-cache-dir -e . --no-deps

# ─── Runtime ──────────────────────────────────────────────────────
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health/live || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
