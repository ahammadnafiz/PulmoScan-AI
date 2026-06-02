#!/usr/bin/env python
"""MLflow Model Registry Registration and Promotion CLI.

Takes a trained MLflow Run, registers it as a Model Version, and compares its
performance against the current "champion" model version. If the new model beats
the champion by the specified margin, it is promoted to the "champion" alias;
otherwise, it is tagged as a "challenger".

Usage:
    python scripts/register_and_promote.py --run-id <run_id>
"""

from __future__ import annotations

import argparse
import sys
import mlflow
from mlflow.exceptions import RestException

from pulmoscan import logger
from pulmoscan.config.configuration import ConfigurationManager
from pulmoscan.monitoring.decisions import should_promote
from pulmoscan.registry import (
    ensure_registered_model,
    get_alias_version,
    get_client,
    get_version_metric,
    register_version,
    set_alias,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="MLflow Model Registry promotion.")
    parser.add_argument("--run-id", required=True, help="The MLflow Run ID containing the model artifact.")
    parser.add_argument("--metric-key", default="accuracy", help="The evaluation metric key to compare (e.g. accuracy, f1).")
    parser.add_argument("--min-margin", type=float, default=0.0, help="Minimum absolute metric improvement needed to promote.")
    args = parser.parse_args()

    config_mgr = ConfigurationManager()
    training_config = config_mgr.get_training_config()

    tracking_uri = training_config.mlflow_tracking_uri
    model_name = training_config.mlflow_experiment_name

    logger.info(f"Connecting to MLflow Tracking Server at {tracking_uri}...")
    client = get_client(tracking_uri)

    # 1. Fetch metric from the target Run
    try:
        run = client.get_run(args.run_id)
        # Search the run metrics for the target key (try test_accuracy first, then key)
        metric_val = run.data.metrics.get(f"test_{args.metric_key}", run.data.metrics.get(args.metric_key))
        if metric_val is None:
            # Fall back to checking best_val_acc from the trainer
            metric_val = run.data.metrics.get("best_val_acc")

        if metric_val is None:
            logger.error(f"Could not find metric '{args.metric_key}' or 'best_val_acc' in run {args.run_id}. Run details: {run.data.metrics}")
            sys.exit(1)
        
        metric_val = float(metric_val)
        logger.info(f"Challenger run '{args.run_id}' metric '{args.metric_key}' = {metric_val:.4f}")
    except RestException as e:
        logger.error(f"Failed to fetch run {args.run_id}: {e}")
        sys.exit(1)

    # 2. Register the run as a new version
    logger.info(f"Registering run '{args.run_id}' under model name '{model_name}'...")
    try:
        version = register_version(
            client,
            name=model_name,
            run_id=args.run_id,
            metrics={args.metric_key: metric_val},
        )
    except Exception as e:
        logger.error(f"Failed to register model version: {e}")
        sys.exit(1)

    # 3. Fetch existing champion
    champion_ver = get_alias_version(client, model_name, "champion")
    champion_metric = None
    if champion_ver is not None:
        champion_metric = get_version_metric(client, model_name, champion_ver.version, args.metric_key)
        logger.info(f"Existing champion version found: v{champion_ver.version} with '{args.metric_key}' = {champion_metric}")
    else:
        logger.info("No active champion found in registry.")

    # 4. Make promotion decision
    decision = should_promote(
        challenger=metric_val,
        champion=champion_metric,
        min_margin=args.min_margin,
    )
    logger.info(f"Promotion decision result: {decision.reason}")

    if decision.promote:
        # If there's an existing champion, we can mark it as a backup challenger
        if champion_ver is not None:
            set_alias(client, model_name, "challenger-backup", champion_ver.version)
        
        # Promote the new model to champion
        set_alias(client, model_name, "champion", version.version)
        logger.info(f"🎉 Promoted model v{version.version} to CHAMPION!")
    else:
        # Tag new model as a challenger
        set_alias(client, model_name, "challenger", version.version)
        logger.info(f"❌ Kept incumbent champion. Challenger version v{version.version} set as CHALLENGER.")


if __name__ == "__main__":
    main()
