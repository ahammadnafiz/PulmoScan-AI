from pathlib import Path

import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from pulmoscan import logger
from pulmoscan.entity.config_entity import EvaluationConfig
from pulmoscan.models import build_model
from pulmoscan.utils.common import get_device, save_json
from pulmoscan.utils.data import build_test_dataset


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.device = get_device()
        self.score: dict | None = None

        # ── MLflow setup ──────────────────────────────────────────────
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
        mlflow.set_experiment(config.mlflow_experiment_name)

    def evaluate(self) -> None:
        """Evaluate the single trained checkpoint on the held-out test set."""
        self.score = self._score([self._load(self.config.trained_model_path)])
        logger.info(f"Evaluation scores: {self.score}")
        self._log_to_mlflow("single-eval")

    def evaluate_ensemble(self, model_paths: list[str]) -> None:
        """Evaluate a fold ensemble (averaged softmax) on the held-out test set."""
        if not model_paths:
            raise ValueError("evaluate_ensemble requires at least one checkpoint.")
        models = [self._load(p) for p in model_paths]
        logger.info(f"Ensemble evaluation: {len(models)} fold checkpoints")
        self.score = self._score(models)
        logger.info(f"Ensemble evaluation scores: {self.score}")
        self._log_to_mlflow("ensemble-eval")

    def _load(self, model_path) -> nn.Module:
        """Build and load a single self-describing checkpoint in eval mode."""
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        model = build_model(
            backbone=checkpoint["backbone"],
            num_classes=checkpoint["num_classes"],
            pretrained=False,
            freeze_backbone=False,
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(self.device)
        model.eval()
        return model

    @torch.no_grad()
    def _score(self, models: list[nn.Module]) -> dict:
        """Run the test set through one or more models, averaging their softmax
        (and the horizontal flip when ``use_tta``), then compute macro metrics."""
        test_ds = build_test_dataset(
            data_root=str(self.config.data_root),
            image_size=self.config.image_size,
        )
        loader = DataLoader(
            test_ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            persistent_workers=self.config.num_workers > 0,
        )

        criterion = nn.CrossEntropyLoss()
        running_loss = 0.0
        all_preds: list[int] = []
        all_labels: list[int] = []

        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            probs = None
            for model in models:
                outputs = model(images)
                # Loss is reported on the first model's logits — a representative
                # scalar; accuracy/F1 below use the averaged ensemble probabilities.
                if probs is None:
                    running_loss += criterion(outputs, labels).item() * images.size(0)
                p = F.softmax(outputs, dim=1)
                if self.config.use_tta:
                    flipped = model(torch.flip(images, dims=[3]))
                    p = (p + F.softmax(flipped, dim=1)) / 2
                probs = p if probs is None else probs + p
            probs = probs / len(models)
            all_preds.extend(probs.argmax(1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

        n = len(loader.dataset)
        accuracy = sum(p == y for p, y in zip(all_preds, all_labels, strict=True)) / n
        return {
            "loss": round(running_loss / n, 6),
            "accuracy": round(accuracy, 6),
            "precision": round(
                precision_score(all_labels, all_preds, average="macro", zero_division=0), 6
            ),
            "recall": round(
                recall_score(all_labels, all_preds, average="macro", zero_division=0), 6
            ),
            "f1": round(f1_score(all_labels, all_preds, average="macro", zero_division=0), 6),
        }

    def save_score(self) -> None:
        if self.score is None:
            raise RuntimeError("Call evaluate() before save_score().")
        save_json(path=Path(self.config.scores_path), data=self.score)

    def _log_to_mlflow(self, run_name: str) -> None:
        """Log evaluation metrics to MLflow."""
        if self.score is None:
            return
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag("stage", "evaluation")
            mlflow.set_tag("eval_mode", run_name)
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {f"test_{k}": v for k, v in self.score.items()}
            )
