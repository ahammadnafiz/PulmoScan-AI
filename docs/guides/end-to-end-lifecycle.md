# 🔄 PulmoScan AI: End-to-End MLOps Lifecycle Playbook

This playbook provides a comprehensive, step-by-step developer guide to reset, execute, monitor, and verify the entire **PulmoScan AI** machine learning lifecycle. It covers everything from absolute clean starts to automated data-drift checks.

---

## 🗺️ MLOps Lifecycle Architecture

```mermaid
flowchart TD
    subgraph Phase 1 [1. Purge & Reset]
        A[Stop Containers] --> B[Wipe local DBs/logs]
        B --> C[Discard dirty artifacts]
    end

    subgraph Phase 2 [2. DVC Pipeline Run]
        D[Data Ingestion] --> E[Prepare Base Model]
        E --> F[CNN Model Training]
        F --> G[Validation Scorecard]
    end

    subgraph Phase 3 [3. MLflow Model Registry]
        H[Boot MLflow + Grafana] --> I[Identify Run UUID]
        I --> J[Register & Tag @champion]
    end

    subgraph Phase 4 [4. Live Serving & Traffic]
        K[Build serving container] --> L[Boot from @champion]
        L --> M[Simulate Scanner Traffic]
    end

    subgraph Phase 5 [5. Drift Gating]
        N[Baseline Reference] --> O[Evidently AI KS-tests]
        O --> P[logs/retrain_decision.json]
    end

    Phase 1 --> Phase 2
    Phase 2 --> Phase 3
    Phase 3 --> Phase 4
    Phase 4 --> Phase 5

    classDef phase fill:#f9f9f9,stroke:#ddd,stroke-width:1px;
    class Phase 1,Phase 2,Phase 3,Phase 4,Phase 5 phase;
```



---

## ⚡ Quick-Start Reference Table


| Step  | Action                 | Command                                     | Purpose                                                               |
| ----- | ---------------------- | ------------------------------------------- | --------------------------------------------------------------------- |
| **1** | **Wipe Environment**   | `make down` + clean scripts                 | Resets volumes, databases, model checkpoints, and logs.               |
| **2** | **Configure Fast-Run** | Edit `params.yaml`                          | Reduces epochs to `1` for high-speed diagnostic validation.           |
| **3** | **Execute Pipeline**   | `dvc repro -f`                              | Ingests data, prepares architecture, trains CNN, and scores test set. |
| **4** | **Start Databases**    | `make obs`                                  | Runs MLflow, Prometheus, and Grafana in the background.               |
| **5** | **Promote Model**      | `python scripts/register_and_promote.py`    | Assigns the `@champion` production alias in the registry.             |
| **6** | **Launch Serving**     | `USE_REGISTRY=true make up`                 | Starts the API, dynamically loading weights from the registry.        |
| **7** | **Simulate Traffic**   | `curl` POST loop                            | Feeds simulated production CT scans into the serving app.             |
| **8** | **Detect Drift**       | `python scripts/check_drift_and_retrain.py` | Compares baseline vs. serving logs to check for distribution shifts.  |


---

## 🚀 Step-by-Step Execution Guide

### Phase 1: Wipe the Environment (Clean Slate)

To start with 100% confidence, we must purge container state, docker volume mounts (which reset Prometheus indexers and Grafana data sources), and delete local logs and cached directories.

Run this block in your terminal:

```bash
# 1. Stop all containers and delete volume cache
docker compose -f docker-compose.yml -f docker-compose.obs.yml down -v

# 2. Delete MLflow sqlite DB, metrics, runs, logs, and local model weights
rm -f mlflow.db
rm -rf mlruns/ logs/ artifacts/ Data/
```

---

### Phase 2: Configure Fast-Track (Diagnostic Verification)

To test this lifecycle quickly (under **60 seconds**) without waiting for the full 35 epochs of deep learning:

1. Open [params.yaml](file:///Users/nafiz/Development/pulmoscan-ai/params.yaml).
2. Change the training epochs configuration to:
  ```yaml
   EPOCHS: 1                  # Set to 1 for diagnostic run
   FINE_TUNE_EPOCHS: 0        # Set to 0 to skip fine-tuning phase
  ```

---

### Phase 3: Execute the DVC Pipeline

Run the full data science workflow. The `-f` (force) flag guarantees DVC bypasses cached hashes and re-runs every stage from absolute scratch:

```bash
dvc repro -f
```

#### What happens during execution?

1. `**data_ingestion**`: Downloads chest CT scan datasets, unpacks files, and splits them into clean subsets: `Data/train`, `Data/valid`, `Data/test`.
2. `**prepare_base_model**`: Instantiates a Transfer Learning framework using `convnext_tiny` with frozen backbones, storing blueprints in `artifacts/prepare_base_model`.
3. `**training**`: Trains the head classification weights on the raw CT images, tracking epochs dynamically and generating the local artifact `artifacts/training/model.pt`.
4. `**evaluation**`: Validates the newly saved weights against unseen `Data/test` splits, writing the scoring output directly to `scores.json`.

---

### Phase 4: Boot MLOps Infrastructure & Register Model

Now we boot the tracking databases and register our new model run as the production-active **Champion**.

#### 1. Start the Observation Stack

This mounts MLflow, Prometheus, and Grafana. MLflow automatically compiles a fresh tracking SQLite schema (`mlflow.db`) upon launching.

```bash
make obs
```

#### 2. Query the Newly Logged Run ID

Retrieve the unique UUID of your new DVC pipeline run directly from your SQLite store:

```bash
sqlite3 mlflow.db "SELECT run_uuid FROM runs LIMIT 1;"
```

*(Copy the resulting character string).*

#### 3. Assign `@champion` Status

Promote your trained checkpoint inside the MLflow Model Registry by running the registration script with your run UUID:

```bash
PYTHONPATH=. python scripts/register_and_promote.py --run-id 0bc0ce3e6b2c415d8ff8d5af26433216 --min-margin 0.0
```

*This binds your PyTorch binary to the `@champion` alias, decoupling your serving code from hard-coded local filepath pathways.*

---

### Phase 5: Containerized Live Serving

Build the API serving Docker container and deploy it, instructing FastAPI to pull its active checkpoints from the Model Registry:

```bash
# 1. Compile the latest FastAPI image
make build

# 2. Launch serving container telling it to load the registry champion
USE_REGISTRY=true make up
```

Verify that the server has loaded your champion weights and is completely healthy:

```bash
curl -fsS http://localhost:8000/api/v1/health/live
curl -fsS http://localhost:8000/api/v1/health/ready
```

---

### Phase 6: Simulate Scanner Traffic

To test live metrics and log prediction attributes, send a stream of requests imitating an active CT-scanner client.

This loop uploads chest scans to the endpoint:

```bash
for i in {1..12}; do
  curl -s -X POST http://localhost:8000/api/v1/predict \
    -F "file=@Data/test/normal/10.png" > /dev/null
done
```

#### Check Real-Time Instrumentation:

Observe the live Prometheus metrics dashboard tracking your image and uncertainty outputs:

```bash
curl -s http://localhost:8000/metrics | grep -E "(prediction|image_brightness|image_contrast|image_sharpness)"
```

You can also view the persistent inference logs recorded under:
[logs/predictions.jsonl](file:///Users/nafiz/Development/pulmoscan-ai/logs/predictions.jsonl)

---

### Phase 7: Run Continuous Data-Drift Detection

Execute the batch CLI script to evaluate distribution changes between your baseline training data and production serving traffic.

```bash
PYTHONPATH=. python scripts/check_drift_and_retrain.py --min-samples 10
```

#### What does this generate?

1. **Dynamic Reference Set**: The script samples `Data/train` to build a baseline feature matrix matching the serving variables.
2. **Evidently AI Distribution Calculations**: Runs statistical tests (Kolmogorov-Smirnov and Population Stability Index) to verify if continuous features (contrast, brightness, sharpness, model confidence, prediction entropy) or categorical labels have drifted.
3. **MLOps Policy Gating**: Saves a clean policy report to `logs/retrain_decision.json` detailing if a self-healing retraining loop must be automatically scheduled.
4. **HTML Report**: Generates an interactive visual report at [logs/drift_report.html](file:///Users/nafiz/Development/pulmoscan-ai/logs/drift_report.html) for manual inspection!

---

## 🛠️ Troubleshooting: Recovering a Corrupted MLflow Database

If you run training locally while the Docker MLflow server is active, concurrent database locks across the macOS Host and Linux container boundary can occasionally corrupt the SQLite database file, resulting in an `sqlalchemy.exc.DatabaseError: (sqlite3.DatabaseError) database disk image is malformed` error.

**Do not panic—you can repair the database and restore all your runs without losing your previous history:**

1. **Stop the MLflow container** to release all active file locks:
  ```bash
   docker compose -f docker-compose.yml -f docker-compose.obs.yml stop mlflow
  ```
2. **Recover the clean database records** using SQLite's native recovery tool:
  ```bash
   sqlite3 mlflow.db ".recover" | sqlite3 mlflow_recovered.db
  ```
   *(Note: Seeing a `defensive off` message is completely normal and means the recovery was successful).*
3. **Swap the databases** (keeping the malformed file as a backup):
  ```bash
   mv mlflow.db mlflow_corrupted.db
   mv mlflow_recovered.db mlflow.db
  ```
4. **Restart the MLOps Observability stack**:
  ```bash
   make obs
  ```
   *Your runs, parameters, metrics, and nested K-Fold fold details are now fully restored and visible at [http://localhost:5050](http://localhost:5050)!*

