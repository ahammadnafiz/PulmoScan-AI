# 🫁 PulmoScan AI

Production-ready chest **CT-scan classifier** (4 classes: adenocarcinoma,
large-cell carcinoma, normal, squamous-cell carcinoma). A PyTorch
transfer-learning CNN (**ConvNeXt-Tiny** by default) trained through a
reproducible **DVC** pipeline and served through a **FastAPI** REST API —
containerized and deployable end to end.

Two training paths produce the same self-describing checkpoint format, so the
serving layer treats them identically:

```
                       ┌──────────────────────────────┐
 DVC pipeline ────────▶│ artifacts/training/model.pt  │──────┐
 ingest → base model   │   single, self-describing    │      │
 → train → evaluate    └──────────────────────────────┘      │
                                                             ├──▶ FastAPI serving
 k-fold script ───────▶┌──────────────────────────────┐      │     /api/v1/predict
 5× stratified CV      │ artifacts/training/folds/    │──────┘     /api/v1/health
 → fold ensemble       │   model_fold0..4.pt          │     (auto-loads the ensemble
                       └──────────────────────────────┘      when present, else single)
```

**Measured (held-out 315-image test set, leakage-audited):** single model + TTA
≈ **91.4%**, 5-fold ensemble + TTA ≈ **92.1%**, with a **95.3% 5-fold
cross-validation** mean as the unbiased generalization estimate. See
[Accuracy & methodology](#accuracy--methodology).

---

## Project layout

```
pulmoscan-ai/
├── app/                          # FastAPI serving application
│   ├── main.py                   # app factory, lifespan (loads model), handlers
│   ├── config.py                 # pydantic-settings (env-driven)
│   ├── api/v1/
│   │   ├── router.py
│   │   └── endpoints/            # health.py, predict.py
│   ├── core/                     # logging, middleware, security (API key)
│   ├── schemas/                  # request/response Pydantic models
│   └── services/inference.py     # loads checkpoint(s) once; single or ensemble
│
├── pulmoscan/                    # Training package (PyTorch + DVC)
│   ├── models/cnn.py             # model factory + transforms (train/serve parity)
│   ├── config/configuration.py   # ConfigurationManager
│   ├── entity/config_entity.py   # typed stage configs (frozen dataclasses)
│   ├── components/               # data_ingestion, prepare_base_model, model_trainer, evaluation
│   ├── pipeline/                 # stage_01..04 (DVC entry points)
│   └── utils/                    # common helpers + dataset/k-fold builder
│
├── config/config.yaml            # paths
├── params.yaml                   # hyperparameters
├── dvc.yaml                      # 4-stage DAG
├── tests/                        # pytest (API)
├── scripts/predict.py            # CLI inference
├── scripts/train_kfold.py        # stratified k-fold training → fold ensemble
├── Dockerfile / docker-compose.yml
└── .github/workflows/ci.yaml     # lint + test
```

A single source of truth — `pulmoscan/models/cnn.py` — defines both the
architecture and the preprocessing transforms, so training and serving can
never drift apart.

---

## Quickstart

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt          # installs torch, app + pulmoscan (editable)
cp .env.example .env
```

### 1. Train (reproducible pipeline)

```bash
dvc repro            # runs only the stages whose inputs changed
# or, without DVC caching:
python main.py
```

Stages: **data ingestion** (gdown download + unzip) → **prepare base model**
(pretrained backbone + fresh head, backbone frozen) → **training** (fine-tune,
save best checkpoint) → **evaluation** (writes `scores.json`).

Tune everything in `params.yaml`:


| Param                                                   | Meaning                                             |
| ------------------------------------------------------- | --------------------------------------------------- |
| `BACKBONE`                                              | `convnext_tiny` (default)                           |
| `PRETRAINED` / `FREEZE_BACKBONE`                        | transfer-learning vs full fine-tune                 |
| `NUM_CLASSES`                                           | validated against the dataset folders at train time |
| `EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`, `WEIGHT_DECAY` | phase 1 (head) training                             |
| `FINE_TUNE_EPOCHS`, `FINE_TUNE_LR`                      | phase 2 (unfrozen backbone) fine-tune               |
| `EARLY_STOPPING_PATIENCE`, `LABEL_SMOOTHING`            | regularization / stopping                           |
| `USE_CLASS_WEIGHTS`, `USE_TTA`                          | class-imbalance weighting · eval-time augmentation  |
| `AUGMENTATION`, `VAL_SPLIT`, `SEED`                     | data                                                |


Device is auto-selected: **CUDA → MPS (Apple Silicon) → CPU**.

### 1b. Train with k-fold cross-validation (→ ensemble)

For a more reliable model-selection signal and a small accuracy bump, train one
model per stratified fold and let the API ensemble them:

```bash
PYTHONPATH=. python scripts/train_kfold.py          # k from params.yaml (K_FOLDS)
```

This pools `Data/train` + `Data/valid`, cuts **stratified** folds, and trains
each from the same prepared base-model init — writing `model_fold0..4.pt` into
`artifacts/training/folds/`. The held-out `Data/test` split is **never read**
during training, so the per-fold validation accuracies are an unbiased
cross-validation estimate and the test set stays a clean generalization probe.
At serving time the API loads every `model_fold*.pt` it finds and averages their
softmax outputs (see [Serve](#2-serve)). Fold count is `K_FOLDS` in `params.yaml`.

Both steps are also wired as DVC stages, so the ensemble path is reproducible:

```bash
dvc repro train_kfold        # → artifacts/training/folds/model_fold*.pt
dvc repro evaluate_ensemble  # → scores_ensemble.json (test-set metrics)
```

`evaluate_ensemble` scores the ensemble on `Data/test` with the same metric
schema as the single-model `evaluation` stage, into `scores_ensemble.json` so
both numbers coexist. (These two stages are heavier than the single-model path —
invoke them by name rather than relying on a blanket `dvc repro`.)

The trained artifacts (`Data/`, `artifacts/`, `models/*.pt`) are gitignored and
stored in a **DVC remote** (Google Drive) — run `dvc pull` after cloning to
fetch them. To set up or reproduce that remote, see
[docs/guides/dvc-gdrive-setup.md](docs/guides/dvc-gdrive-setup.md).

### 2. Serve

```bash
uvicorn app.main:app --reload --port 8000
```

**Model loading is automatic** — at startup the API:

1. loads the **fold ensemble** if `ENSEMBLE_DIR` (default `artifacts/training/folds`)
  holds ≥ 2 `model_fold*.pt` files, averaging their softmax; otherwise
2. loads the **single** checkpoint at `MODEL_PATH`; otherwise
3. stays up in **degraded** mode (predictions return 503).

Test-time augmentation (`USE_TTA`) is **on by default** for serving — each model
also scores the horizontal flip and the two are averaged. To serve the single
model instead of the ensemble, point `ENSEMBLE_DIR` at an empty/absent path.

- Swagger docs: [http://localhost:8000/docs](http://localhost:8000/docs)
- Health: `GET /api/v1/health` · liveness `/health/live` · readiness `/health/ready`

```bash
# predict from an uploaded image
curl -X POST http://localhost:8000/api/v1/predict \
  -F "file=@scan.png"

# list the model's classes
curl http://localhost:8000/api/v1/predict/classes
```

Or from the CLI without the server:

```bash
python scripts/predict.py path/to/scan.png
```

### 3. Test & lint

```bash
pytest          # API tests (model is mocked — no checkpoint needed)
ruff check .
```

---

## Docker

A **multi-stage** build produces a self-contained, deployable image (~1.9 GB,
mostly CPU-only PyTorch). The runtime stage installs **serving deps only**
(`requirements-serve.txt` — no DVC/scikit-learn/test tooling), copies a prebuilt
venv, runs as a **non-root** user, and has a healthcheck on `/api/v1/health/live`.

```bash
# bake the trained checkpoint into the image, then build + run
cp artifacts/training/model.pt models/model.pt   # if not already there
docker compose up --build
```

The single model is **baked into the image** (`MODEL_PATH=/app/models/model.pt`),
so the container is self-contained — push it to any registry and run it anywhere,
no volume required. It serves ~91% (single model + TTA) on CPU.

**Serve the 5-fold ensemble** (~92%) instead — no rebuild needed, just mount the
folds (the image's `ENSEMBLE_DIR` already points at `/app/models/folds`, so the
API auto-detects and averages them):

```bash
docker run -p 8000:8000 \
  -v "$PWD/artifacts/training/folds:/app/models/folds:ro" \
  pulmoscan-ai:latest
```

(or uncomment the folds volume in `docker-compose.yml`).

**Runtime env** (see [Configuration](#configuration-env-vars)): `WORKERS`
(default 1 — one model in memory; scale out with replicas, not workers),
`USE_TTA`, `API_KEY`, `CORS_ORIGINS`.

### Published image (GHCR)

The `[docker-publish](.github/workflows/docker-publish.yml)` workflow builds and
pushes the image to **GitHub Container Registry** on every `v*.*.`* tag and
published release:

```bash
docker run -p 8000:8000 ghcr.io/ahammadnafiz/pulmoscan-ai:latest
```

**Release process** — the checkpoint is gitignored (it lives in DVC/Drive), so
CI can't see it. To publish a *self-contained* image, attach `model.pt` to the
GitHub release: the workflow downloads that asset and bakes it in. Without the
asset it builds a **model-less** image (mount a model at `/app/models` at run
time). Tag a release:

```bash
git tag v1.0.0 && git push origin v1.0.0
# then attach artifacts/training/model.pt to the release on GitHub
```

---

## Configuration (env vars)


| Var                | Default                       | Purpose                                                                |
| ------------------ | ----------------------------- | ---------------------------------------------------------------------- |
| `ENSEMBLE_DIR`     | `artifacts/training/folds`    | dir of `model_fold*.pt`; ≥2 → ensemble, else fall back to `MODEL_PATH` |
| `MODEL_PATH`       | `artifacts/training/model.pt` | single checkpoint used when no ensemble is found                       |
| `USE_TTA`          | `true`                        | average over the horizontal flip at inference                          |
| `API_KEY`          | *(empty)*                     | when set, prediction routes require `X-API-Key`                        |
| `CORS_ORIGINS`     | `http://localhost:3000`       | comma-separated allowed origins                                        |
| `MAX_UPLOAD_BYTES` | `10485760`                    | reject larger uploads (413)                                            |
| `LOG_LEVEL`        | `INFO`                        | loguru level                                                           |


Copy `.env.example` → `.env` to get these defaults. Out of the box the API
serves the **fold ensemble + TTA** (the higher-accuracy mode); set `ENSEMBLE_DIR=`
to fall back to the single `MODEL_PATH` checkpoint.

If no checkpoint is found at startup the API stays up in **degraded** mode
(`/health` reports `model_loaded: false`, predictions return 503) rather than
crashing.

---

## Accuracy & methodology

All numbers below are measured on the **held-out 315-image `Data/test` split**,
which is *never* seen during training or model selection.


| Configuration         | Test accuracy | Macro F1 | Median latency | Metrics file           |
| --------------------- | ------------- | -------- | -------------- | ---------------------- |
| Single model + TTA    | 91.4%         | 0.919    | ~14 ms         | `scores.json`          |
| 5-fold ensemble + TTA | 92.1%         | 0.925    | ~57 ms         | `scores_ensemble.json` |


(Reproduce with `dvc repro evaluation` and `dvc repro evaluate_ensemble`.)

The honest generalization estimate is the **5-fold cross-validation mean of
~95.3%** (per-fold range 94.2–97.1%), computed on the pooled `train+valid` data
only. The ~3-point gap between CV (95.3%) and test (92.1%) reflects a genuine
distribution shift between the train/valid and test splits of this dataset — it
is a data-quality ceiling, not a tuning deficit.

Why these choices avoid optimistic bias:

- **Stratified k-fold** replaces the original fixed 72-image validation set
(where a single image was worth ~1.4%) with 5 × 137-image folds, so
checkpoint selection is no longer a coin-flip.
- **The test set is touched exactly once**, for final reporting — no
hyperparameter was tuned against it, so there is no leakage.
- **TTA and ensembling are label-agnostic** (flip-averaging and softmax-averaging
never consult the ground truth), so they reduce variance without biasing
toward the test distribution.

Treat ~92% test / ~95% CV as the defensible figure; claims of 98–99% on this
particular dataset usually indicate tuning against the test set or train/test
contamination.

---

## Notes

- Every checkpoint — the single `model.pt` and each `model_fold*.pt` — is
**self-describing**: it stores the backbone, class names, and image size, so the
API rebuilds the exact model with no external metadata. Ensemble members are
checked for matching architecture/labels/image-size before their probabilities
are averaged.
- All `torch.load` calls use `weights_only=True` (no arbitrary-code-execution
risk from untrusted checkpoints).
- The dataset's class names come straight from the `ImageFolder` directory
names — no labels are hardcoded. Train/valid folders carry TNM-staging suffixes
(e.g. `adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib`) that are normalized to the
canonical class so every split shares one label space.

