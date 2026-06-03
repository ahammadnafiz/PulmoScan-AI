# 🫁 Testing Data Drift in PulmoScan AI

This guide walks you through how **data drift detection** is architected in PulmoScan AI, and provides a step-by-step tutorial on how to simulate and test drift using custom out-of-distribution (OOD) scans.

---

## 1. How Drift Detection Works

In a medical context, **data drift** (or covariate shift) occurs when the input distribution of scans in production changes relative to the dataset the model was trained on. This can be caused by:
*   Different CT scanner manufacturers or models.
*   Variations in scanner resolution, brightness, or contrast settings.
*   Changes in image prep/compression software.
*   Subtle shifts in patient demographics (e.g., higher severity cases).

### The Feature Vector
Because statistical tests fail on raw image pixel spaces, the system extracts **interpretable image-quality metrics** alongside **model-confidence signals** to represent each prediction:

| Metric Category | Feature Name | Description |
| :--- | :--- | :--- |
| **Geometry** | `width`, `height`, `aspect_ratio` | Dimensions of the uploaded scan |
| **Intensity** | `brightness_mean`, `contrast_std` | Normalized mean and standard deviation of grayscale pixels |
| **Quality** | `sharpness` | Variance of a 4-neighbour Laplacian (focus proxy) |
| **Histograms** | `dark_fraction`, `bright_fraction` | Percentage of pixels near absolute black (<0.1) or white (>0.9) |
| **Model Output** | `predicted_label`, `confidence`, `entropy` | Softmax predictions, confidence, and Shannon uncertainty |

### The Logging Flow

```
                      ┌──────────────────────┐
                      │ API Request (Scan)   │
                      └──────────┬───────────┘
                                 │
                                 ▼
                     [ inference_service.predict ]
                                 │
          ┌──────────────────────┴──────────────────────┐
          ▼                                             ▼
[ Extract Quality Feats ]                      [ PyTorch Inference ]
  - brightness, sharpness                        - label, confidence, entropy
          │                                             │
          └──────────────────────┬──────────────────────┘
                                 │
                                 ▼
                     [ Append json line ]
                     logs/predictions.jsonl
```

---

## 2. Step-by-Step: Simulating and Testing Drift

Follow these steps to generate baseline logs, inject drifted OOD scans, run the detector, and review the alerts.

### Step A: Start the API Server
First, spin up the FastAPI service locally:
```bash
uvicorn app.main:app --reload --port 8000
```
Verify that the model loads successfully (or fallback to single model mode). You can hit `http://localhost:8000/api/v1/health` to confirm `model_loaded: true`.

---

### Step B: Create a Python Simulation Script
Create a temporary script in `scratch/simulate_drift.py` that loads a clean image from `Data/test/normal/`, applies modifications (such as extreme brightness or blur) to simulate scanner variations, and uploads them to the server.

Let's write a script that sends:
1.  **5 clean requests** (Normal scanner behavior).
2.  **10 drifted requests** (Simulating a miscalibrated scanner with high contrast and blur).

```python
import io
import time
import requests
from PIL import Image, ImageEnhance, ImageFilter

API_URL = "http://localhost:8000/api/v1/predict"
# Ensure you have the test file path
clean_img_path = "Data/test/normal/normal.png"

def send_image(img):
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    files = {"file": ("scan.png", buffer, "image/png")}
    response = requests.post(API_URL, files=files)
    print(f"Status: {response.status_code} | Pred: {response.json().get('label')} | Conf: {response.json().get('confidence')}")

# 1. Load clean test image
base_image = Image.open(clean_img_path)

# --- Phase 1: Send Normal Scans ---
print("Sending normal scans...")
for _ in range(5):
    send_image(base_image)
    time.sleep(0.1)

# --- Phase 2: Send Drifted Scans (Highly blurred and brightened) ---
print("\nSending out-of-distribution (drifted) scans...")
# Enhance brightness by 2x and apply Gaussian Blur
drifted_image = ImageEnhance.Brightness(base_image).enhance(2.0)
drifted_image = drifted_image.filter(ImageFilter.GaussianBlur(radius=5))

for _ in range(10):
    send_image(drifted_image)
    time.sleep(0.1)
```

---

### Step C: Execute the Simulation
Run your simulation script to write logs to `logs/predictions.jsonl`:
```bash
python scratch/simulate_drift.py
```
This adds 15 prediction records containing both normal scans and highly brightened, blurred scans.

---

### Step D: Trigger the Drift Check
Execute the drift analyzer. We set `--min-samples 10` so that it processes the 15 production logs we just generated, and we write an interactive HTML report to `logs/drift_report.html`:

```bash
PYTHONPATH=. python scripts/check_drift_and_retrain.py --min-samples 10 --html-report logs/drift_report.html
```

---

## 3. Evaluating the Outputs

The drift script produces two output artifacts:

### 1. The Gating Decision: `logs/retrain_decision.json`
This metadata file stores the automated policy evaluation. If the drift share exceeds `0.50` (50% of quality columns drifted), it recommends retraining:

```json
{
  "retrain": true,
  "reasons": [
    "Evidently reported dataset drift",
    "drift share 0.64 >= threshold 0.50"
  ],
  "low_confidence_rate": 0.208,
  "dataset_drift": true,
  "drift_share": 0.636,
  "n_drifted_columns": 7,
  "n_columns": 11,
  "drifted_columns": [
    "brightness_mean",
    "sharpness",
    "dark_fraction",
    "bright_fraction",
    "confidence",
    "entropy"
  ]
}
```

### 2. The Interactive Report: `logs/drift_report.html`
Open the generated HTML report in your browser to inspect interactive graphs.
*   **Drift Detection Summary**: Shows a high-level summary of columns and their corresponding statistical tests.
*   **Column Distributions**: Offers side-by-side Histograms and density plots comparing the training reference distribution with your 15 new production queries. You will see the `sharpness` values shifted towards zero (due to blur) and `brightness_mean` shifted right (due to brightness enhancement).

---

## 4. Retraining Automation

In production environments, the output of `logs/retrain_decision.json` can be checked by cron jobs or pipeline schedulers.
*   If `retrain` evaluates to `true`, the scheduler can automatically trigger:
    ```bash
    dvc repro
    ```
    This fetches the latest dataset updates, recreates base models, runs training and validation, and registers a new candidate version to the MLflow Model Registry.
