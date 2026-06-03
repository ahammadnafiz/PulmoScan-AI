#!/usr/bin/env python
"""Simulate Production Traffic and Test Data Drift Gating.

This script:
1. Resets the production logs file (logs/predictions.jsonl).
2. Runs a series of clean test images through the model to simulate normal production traffic.
3. Runs modified (blurred, high-brightness, or resized) images to simulate scanner drift.
4. Executes the drift report analysis and checks the MLOps retraining policy.

Usage:
    python scripts/simulate_drift.py --n-clean 15 --n-drifted 20
"""

from __future__ import annotations

import argparse
import glob
import io
import os
import subprocess
import sys

from PIL import Image, ImageEnhance, ImageFilter

from app.services.inference import inference_service
from pulmoscan import logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate production traffic and test drift detection.")
    parser.add_argument(
        "--model",
        default="artifacts/training/model.pt",
        help="Path to the model checkpoint to load.",
    )
    parser.add_argument(
        "--n-clean",
        type=int,
        default=15,
        help="Number of normal (clean) scans to simulate.",
    )
    parser.add_argument(
        "--n-drifted",
        type=int,
        default=25,
        help="Number of drifted (OOD) scans to simulate.",
    )
    parser.add_argument(
        "--test-dir",
        default="Data/test",
        help="Directory containing the test images.",
    )
    return parser.parse_args()


def get_test_images(test_dir: str) -> list[str]:
    """Gather all image paths inside the test directory."""
    patterns = [os.path.join(test_dir, "**", "*.png"), os.path.join(test_dir, "**", "*.jpg")]
    images = []
    for pattern in patterns:
        images.extend(glob.glob(pattern, recursive=True))
    if not images:
        logger.error(f"No test images found in {test_dir}. Make sure you run DVC ingestion first.")
        sys.exit(1)
    return sorted(images)


def apply_drift(img: Image.Image) -> Image.Image:
    """Simulate scanner drift by applying heavy blur and increasing brightness."""
    # Apply Gaussian blur to simulate focus/resolution issues
    drifted = img.filter(ImageFilter.GaussianBlur(radius=4.0))
    # Apply brightness enhancement to simulate scanner calibration issues (2.2x brightness)
    drifted = ImageEnhance.Brightness(drifted).enhance(2.2)
    return drifted


def main() -> None:
    args = parse_args()

    # 1. Reset prediction logs to make the test clean and deterministic
    log_path = "logs/predictions.jsonl"
    logger.info(f"Resetting prediction log file at '{log_path}'...")
    os.makedirs("logs", exist_ok=True)
    with open(log_path, "w") as f:
        f.truncate(0)

    # 2. Load model into inference service
    if not os.path.exists(args.model):
        logger.error(f"Model not found at '{args.model}'. Please run 'dvc repro' or training first.")
        sys.exit(1)
    
    logger.info(f"Loading checkpoint '{args.model}'...")
    inference_service.load(args.model)

    # 3. Retrieve test scans
    image_paths = get_test_images(args.test_dir)
    logger.info(f"Found {len(image_paths)} test scans to use for simulation.")

    # 4. Simulate clean traffic
    n_clean = min(args.n_clean, len(image_paths))
    logger.info(f"--- Simulating Phase 1: {n_clean} Clean Scans ---")
    for i in range(n_clean):
        img_path = image_paths[i]
        try:
            with Image.open(img_path) as img:
                img_rgb = img.convert("RGB")
                # Save to bytes
                buf = io.BytesIO()
                img_rgb.save(buf, format="PNG")
                # Perform inference (this automatically logs to logs/predictions.jsonl)
                res = inference_service.predict(buf.getvalue())
                logger.info(f"Clean [{i+1}/{n_clean}]: {os.path.basename(img_path)} -> Predict: {res['label']} (Conf: {res['confidence']})")
        except Exception as e:
            logger.error(f"Failed simulating clean image {img_path}: {e}")

    # 5. Simulate drifted traffic
    n_drifted = min(args.n_drifted, len(image_paths))
    logger.info(f"--- Simulating Phase 2: {n_drifted} Drifted (OOD) Scans ---")
    # Let's use different images for drift simulation to ensure variance
    start_idx = len(image_paths) - n_drifted
    for i in range(n_drifted):
        img_path = image_paths[start_idx + i]
        try:
            with Image.open(img_path) as img:
                img_rgb = img.convert("RGB")
                # Apply simulated scanner drift
                drifted_img = apply_drift(img_rgb)
                
                # Save to bytes
                buf = io.BytesIO()
                drifted_img.save(buf, format="PNG")
                # Perform inference
                res = inference_service.predict(buf.getvalue())
                logger.info(f"Drifted [{i+1}/{n_drifted}]: {os.path.basename(img_path)} -> Predict: {res['label']} (Conf: {res['confidence']})")
        except Exception as e:
            logger.error(f"Failed simulating drifted image {img_path}: {e}")

    # 6. Run the drift report script to evaluate policy decisions
    logger.info("--- Phase 3: Running Drift Report and Gating Policy ---")
    
    # Run scripts/check_drift_and_retrain.py via subprocess
    cmd = [sys.executable, "scripts/check_drift_and_retrain.py", "--min-samples", str(n_clean + n_drifted)]
    logger.info(f"Running command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    # Output decision location
    decision_path = "logs/retrain_decision.json"
    if os.path.exists(decision_path):
        with open(decision_path) as f:
            decision = f.read()
        logger.info(f"Saved Retrain Decision output ({decision_path}):\n{decision}")


if __name__ == "__main__":
    main()
