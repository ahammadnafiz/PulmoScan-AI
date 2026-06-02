"""Scalar image + prediction features for drift monitoring.

Evidently (and tabular drift detection in general) operates on columns, not raw
pixels. So to monitor a CT-image classifier we reduce each image to a handful of
interpretable scalars — brightness, contrast, sharpness, geometry — and pair
them with the model's own output signals (predicted class, confidence, entropy).
Drift in *these* columns is what flags a new scanner, a preprocessing change, or
a stream of out-of-distribution inputs.

Pure PIL + numpy: this module is imported both by the serving container and the
offline reference builder, so the production and reference feature spaces are
guaranteed identical.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from PIL import Image

# The canonical image-feature column order. Reference tables and per-request
# logs both use these names so Evidently lines the two frames up column-for-column.
FEATURE_COLUMNS: list[str] = [
    "width",
    "height",
    "aspect_ratio",
    "brightness_mean",
    "contrast_std",
    "sharpness",
    "dark_fraction",
    "bright_fraction",
]


def extract_image_features(image: Image.Image) -> dict[str, float]:
    """Reduce a PIL image to the scalar quality features used for drift checks.

    All intensity features are computed on a 0..1 normalized grayscale view so
    they are independent of bit depth. ``sharpness`` is the variance of a 4-neighbour
    Laplacian (edge energy) — a cheap proxy for focus/resolution that shifts hard
    when inputs are blurred, upscaled, or come from a different acquisition.
    """
    width, height = image.size
    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0

    if gray.shape[0] > 2 and gray.shape[1] > 2:
        center = gray[1:-1, 1:-1]
        laplacian = (
            gray[:-2, 1:-1]
            + gray[2:, 1:-1]
            + gray[1:-1, :-2]
            + gray[1:-1, 2:]
            - 4.0 * center
        )
        sharpness = float(laplacian.var())
    else:
        sharpness = 0.0

    return {
        "width": float(width),
        "height": float(height),
        "aspect_ratio": float(width / height) if height else 0.0,
        "brightness_mean": float(gray.mean()),
        "contrast_std": float(gray.std()),
        "sharpness": sharpness,
        "dark_fraction": float((gray < 0.1).mean()),
        "bright_fraction": float((gray > 0.9).mean()),
    }


def softmax_entropy(probabilities: Sequence[float]) -> float:
    """Shannon entropy (nats) of a probability vector.

    High entropy means the model spread its mass across classes — i.e. it was
    uncertain. A sustained rise in mean entropy is a classic out-of-distribution
    signal even when accuracy can't be measured (no labels in production).
    """
    return float(-sum(p * math.log(p) for p in probabilities if p > 0.0))


def normalized_entropy(probabilities: Sequence[float]) -> float:
    """Entropy scaled to 0..1 (divided by log(num_classes)) so it is comparable
    across models with different class counts. 0 = fully confident, 1 = uniform."""
    n = len(probabilities)
    if n <= 1:
        return 0.0
    return softmax_entropy(probabilities) / math.log(n)
