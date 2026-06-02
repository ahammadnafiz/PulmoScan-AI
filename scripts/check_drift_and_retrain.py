#!/usr/bin/env python
"""MLOps Drift Detection and Retrain Gating CLI.

Builds a reference baseline from the training set, extracts production features
from logs/predictions.jsonl, runs Evidently to check for data and prediction
distribution drift, and evaluates MLOps policies to decide if retraining is required.

Usage:
    python scripts/check_drift_and_retrain.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image

from pulmoscan import logger
from pulmoscan.config.configuration import ConfigurationManager
from pulmoscan.models.cnn import get_eval_transforms
from pulmoscan.monitoring import extract_image_features, normalized_entropy
from pulmoscan.monitoring.decisions import decide_retrain
from pulmoscan.monitoring.drift import run_drift_report
from pulmoscan.utils.common import get_device
from pulmoscan.utils.data import build_datasets


def load_model(model_path: str, device) -> torch.nn.Module:
    """Load the trained model from self-describing checkpoint."""
    from pulmoscan.models import build_model
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model = build_model(
        backbone=checkpoint["backbone"],
        num_classes=checkpoint["num_classes"],
        pretrained=False,
        freeze_backbone=False,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model


def build_reference_dataset(model_path: str, data_root: str, n_samples: int = 100) -> pd.DataFrame:
    """Build a reference baseline DataFrame from a subset of training data.
    
    Extracts image quality features and runs the model to get baseline
    confidences, predictions, and entropy.
    """
    logger.info(f"Building reference baseline of {n_samples} samples from training data...")
    device = get_device()
    
    # Load model and configurations
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model = load_model(model_path, device)
    class_names = checkpoint["class_names"]
    image_size = checkpoint["image_size"]
    
    # Load dataset
    train_ds, _, _, _ = build_datasets(
        data_root=data_root,
        image_size=image_size,
        augmentation=False,
        val_split=0.2,
        seed=42
    )
    
    # Take a random subset of size n_samples
    indices = np.random.choice(len(train_ds), min(n_samples, len(train_ds)), replace=False)
    
    records = []
    transforms = get_eval_transforms(image_size)
    
    for idx in indices:
        # Load raw image for feature extraction
        img_path, label_idx = train_ds.samples[idx]
        image = Image.open(img_path).convert("RGB")
        features = extract_image_features(image)
        
        # Run through model for predictions
        tensor = transforms(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1)[0].cpu().tolist()
            
        top_idx = int(np.argmax(probs))
        entropy = normalized_entropy(probs)
        
        record = {
            "predicted_label": class_names[top_idx],
            "confidence": probs[top_idx],
            "entropy": entropy,
            **features
        }
        records.append(record)
        
    return pd.DataFrame(records)


def load_production_dataset(log_path: str) -> pd.DataFrame:
    """Load logged production predictions from JSONL."""
    if not os.path.exists(log_path):
        return pd.DataFrame()
        
    records = []
    with open(log_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Data drift and retrain detection.")
    parser.add_argument("--reference-samples", type=int, default=100, help="Number of baseline training samples.")
    parser.add_argument("--min-samples", type=int, default=10, help="Minimum production samples required to check drift.")
    parser.add_argument("--html-report", default="logs/drift_report.html", help="Path to write the Evidently HTML report.")
    args = parser.parse_args()

    config_mgr = ConfigurationManager()
    training_config = config_mgr.get_training_config()
    
    model_path = str(training_config.trained_model_path)
    data_root = str(training_config.data_root)
    log_path = "logs/predictions.jsonl"
    
    if not os.path.exists(model_path):
        logger.error(f"Trained model not found at {model_path}. Run training first.")
        sys.exit(1)

    # 1. Load production data
    logger.info(f"Reading production logs from {log_path}...")
    prod_df = load_production_dataset(log_path)
    logger.info(f"Found {len(prod_df)} production predictions.")
    
    if len(prod_df) < args.min_samples:
        logger.warning(f"Abstaining from drift check: insufficient production samples ({len(prod_df)} < {args.min_samples})")
        sys.exit(0)

    # 2. Build reference baseline
    ref_df = build_reference_dataset(model_path, data_root, args.reference_samples)
    
    # 3. Run Evidently Drift Report
    logger.info("Running Evidently drift report...")
    os.makedirs(os.path.dirname(args.html_report), exist_ok=True)
    summary, raw = run_drift_report(ref_df, prod_df, html_path=args.html_report)
    
    logger.info(f"Drift Summary: Dataset Drifted = {summary.dataset_drift} | Drift Share = {summary.drift_share:.2f} ({summary.n_drifted_columns}/{summary.n_columns} columns drifted)")
    if summary.drifted_columns:
        logger.info(f"Drifted Columns: {summary.drifted_columns}")
    logger.info(f"HTML report saved to {args.html_report}")

    # 4. Make Retraining Decision
    # A low confidence prediction has confidence < 0.7
    low_conf_mask = prod_df["confidence"] < 0.7
    low_confidence_rate = float(low_conf_mask.mean())
    logger.info(f"Production Low-Confidence Rate (<70%): {low_confidence_rate:.2f} ({low_conf_mask.sum()}/{len(prod_df)} samples)")
    
    decision = decide_retrain(
        dataset_drift=summary.dataset_drift,
        drift_share=summary.drift_share,
        low_confidence_rate=low_confidence_rate,
        n_current_samples=len(prod_df),
        min_samples=args.min_samples,
    )
    
    logger.info(f"Retraining Decision: RETRAIN = {decision.retrain}")
    for reason in decision.reasons:
        logger.info(f"  - Trigger: {reason}")
        
    # Write decision to a json artifact for CI/CD gates
    decision_artifact = "logs/retrain_decision.json"
    with open(decision_artifact, "w") as f:
        json.dump({
            "retrain": decision.retrain,
            "reasons": decision.reasons,
            "low_confidence_rate": low_confidence_rate,
            **summary.as_dict()
        }, f, indent=2)
    logger.info(f"Retraining decision metadata saved to {decision_artifact}")


if __name__ == "__main__":
    main()
