"""Inference service: loads the trained checkpoint once and serves predictions.

The model is loaded at application startup (lifespan) and cached in memory —
loading per request would make the API unusable. Preprocessing reuses the
exact transforms from the training package, guaranteeing train/serve parity.
"""

from __future__ import annotations

import io
import time

import torch
import torch.nn.functional as F
from loguru import logger
from PIL import Image

from pulmoscan.models import build_model, get_eval_transforms
from pulmoscan.utils.common import get_device


class InferenceService:
    def __init__(self) -> None:
        self._models: list[torch.nn.Module] = []
        self._class_names: list[str] = []
        self._backbone: str = ""
        self._image_size: int = 224
        self._device = get_device()
        self._transforms = None
        self.use_tta: bool = False

    @property
    def is_ready(self) -> bool:
        return len(self._models) > 0

    @property
    def num_models(self) -> int:
        return len(self._models)

    @property
    def class_names(self) -> list[str]:
        return self._class_names

    @property
    def backbone(self) -> str:
        return self._backbone

    @property
    def image_size(self) -> int:
        return self._image_size

    @property
    def device(self) -> str:
        return str(self._device)

    def _load_one(self, model_path: str) -> torch.nn.Module:
        """Build and load a single self-describing checkpoint onto the device."""
        checkpoint = torch.load(model_path, map_location=self._device, weights_only=True)
        backbone = checkpoint["backbone"]
        class_names = list(checkpoint["class_names"])
        image_size = int(checkpoint["image_size"])

        # All ensemble members must share architecture/labels/preprocessing —
        # otherwise averaging their probabilities is meaningless.
        if self._models:
            if (backbone, class_names, image_size) != (
                self._backbone,
                self._class_names,
                self._image_size,
            ):
                raise ValueError(
                    f"Checkpoint {model_path} is incompatible with the loaded ensemble "
                    f"(backbone/classes/image_size mismatch)."
                )
        else:
            self._backbone, self._class_names, self._image_size = backbone, class_names, image_size
            self._transforms = get_eval_transforms(image_size)

        model = build_model(
            backbone=backbone,
            num_classes=checkpoint["num_classes"],
            pretrained=False,
            freeze_backbone=False,
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(self._device)
        model.eval()
        return model

    def load(self, model_path: str) -> None:
        """Load a single self-describing checkpoint, replacing any loaded models."""
        self._models = []
        self._models.append(self._load_one(model_path))
        logger.info(
            f"Model loaded: backbone={self._backbone}, classes={self._class_names}, "
            f"device={self._device}"
        )

    def load_ensemble(self, model_paths: list[str]) -> None:
        """Load several fold checkpoints as an ensemble (averaged softmax)."""
        if not model_paths:
            raise ValueError("load_ensemble requires at least one checkpoint path.")
        self._models = []
        for path in model_paths:
            self._models.append(self._load_one(path))
        logger.info(
            f"Ensemble loaded: {len(self._models)} models, backbone={self._backbone}, "
            f"classes={self._class_names}, device={self._device}"
        )

    @torch.no_grad()
    def predict(self, image_bytes: bytes) -> dict:
        """Run inference on raw image bytes and return label + probabilities.

        Probabilities are averaged across all loaded models (ensemble) and, when
        ``use_tta`` is on, across the horizontal flip of each — both reduce
        variance without ever consulting the labels, so neither biases the result.
        """
        if not self._models or self._transforms is None:
            raise RuntimeError("Model is not loaded.")

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:  # noqa: BLE001 — surface a clean client error
            raise ValueError(f"Could not decode image: {e}") from e

        tensor = self._transforms(image).unsqueeze(0).to(self._device)

        start = time.time()
        probs = None
        for model in self._models:
            p = F.softmax(model(tensor), dim=1)
            if self.use_tta:
                flipped = F.softmax(model(torch.flip(tensor, dims=[3])), dim=1)
                p = (p + flipped) / 2
            probs = p if probs is None else probs + p
        probs = (probs / len(self._models))[0].cpu()
        inference_ms = (time.time() - start) * 1000

        top_idx = int(probs.argmax().item())
        return {
            "label": self._class_names[top_idx],
            "confidence": round(float(probs[top_idx]), 6),
            "probabilities": [
                {"label": name, "probability": round(float(p), 6)}
                for name, p in zip(self._class_names, probs.tolist(), strict=True)
            ],
            "inference_time_ms": round(inference_ms, 3),
        }


# Singleton — one model in memory for the whole process.
inference_service = InferenceService()
