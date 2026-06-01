"""Stratified k-fold training → fold checkpoints for ensembling.

Trains one model per fold from the same prepared base-model init and writes
``model_fold{i}.pt`` into ``artifacts/training/folds/``. The serving API picks
these up automatically (see ``app.config.Settings.ensemble_dir``) and averages
their softmax outputs.

Leakage-free by construction: folds are cut only from the pooled train+valid
data; the held-out ``Data/test`` split is never read here. The per-fold
validation accuracies are therefore an unbiased cross-validation estimate.

Usage:
    PYTHONPATH=. python scripts/train_kfold.py          # k from params.yaml (K_FOLDS)
    PYTHONPATH=. python scripts/train_kfold.py --k 5    # explicit override
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pulmoscan import logger
from pulmoscan.components.model_trainer import Training
from pulmoscan.config.configuration import ConfigurationManager

STAGE_NAME = "K-Fold Training stage"


def main(k: int | None) -> None:
    config = ConfigurationManager()
    training_config = config.get_training_config()
    folds = k if k is not None else training_config.k_folds
    out_dir = Path(training_config.trained_model_path).parent / "folds"

    training = Training(config=training_config)
    fold_scores = training.train_kfold(k=folds, out_dir=out_dir)

    mean = sum(fold_scores) / len(fold_scores)
    logger.info(
        f"Saved {len(fold_scores)} fold checkpoints to {out_dir} | CV mean val_acc={mean:.4f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stratified k-fold training.")
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Number of folds (default: K_FOLDS from params.yaml).",
    )
    args = parser.parse_args()

    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started (k={args.k or 'params'}) <<<<<<")
        main(args.k)
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
