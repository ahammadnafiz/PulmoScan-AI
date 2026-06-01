"""Evaluate the fold ensemble on the held-out test set → ``scores_ensemble.json``.

Loads every ``model_fold*.pt`` produced by ``scripts/train_kfold.py``, averages
their softmax (plus the horizontal flip when ``USE_TTA`` is on), and writes the
same metric schema as the single-model evaluation stage — but to
``ensemble_scores_path`` so both sets of numbers coexist.

This is the DVC ``evaluate_ensemble`` stage. The test set is read here only for
final scoring; it never participates in training or fold selection.

Usage:
    PYTHONPATH=. python scripts/eval_ensemble.py
"""

from __future__ import annotations

import glob
from pathlib import Path

from pulmoscan import logger
from pulmoscan.components.evaluation import Evaluation
from pulmoscan.config.configuration import ConfigurationManager

STAGE_NAME = "Ensemble Evaluation stage"


def main() -> None:
    config = ConfigurationManager()
    eval_config = config.get_ensemble_evaluation_config()
    training_config = config.get_training_config()

    folds_dir = Path(training_config.trained_model_path).parent / "folds"
    model_paths = sorted(glob.glob(str(folds_dir / "model_fold*.pt")))
    if not model_paths:
        raise FileNotFoundError(
            f"No fold checkpoints in {folds_dir}. Run scripts/train_kfold.py first."
        )

    evaluation = Evaluation(config=eval_config)
    evaluation.evaluate_ensemble(model_paths)
    evaluation.save_score()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
