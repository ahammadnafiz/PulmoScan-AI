"""Grad-CAM math: verify the hook + attribution produce a valid heatmap.

Uses a freshly built (untrained, non-pretrained) ConvNeXt-Tiny so the test runs
offline and fast — we are checking the mechanism (target-layer hook, gradient
weighting, normalization, overlay), not learned localization quality.
"""

import torch
from PIL import Image

from app.services.explain import (
    grad_cam,
    overlay_on_image,
    resolve_target_layer,
)
from pulmoscan.models import build_model


def test_grad_cam_produces_normalized_map():
    model = build_model(
        backbone="convnext_tiny", num_classes=4, pretrained=False, freeze_backbone=False
    )
    model.eval()
    target_layer = resolve_target_layer(model, "convnext_tiny")
    tensor = torch.randn(1, 3, 224, 224)

    cam, probs = grad_cam(model, target_layer, tensor)

    assert cam.ndim == 2  # [H, W]
    assert cam.min() >= 0.0 and cam.max() <= 1.0  # normalized to 0..1
    assert probs.shape == (4,)
    assert abs(float(probs.sum()) - 1.0) < 1e-4  # softmax sums to 1


def test_overlay_matches_input_size():
    model = build_model(
        backbone="convnext_tiny", num_classes=4, pretrained=False, freeze_backbone=False
    )
    model.eval()
    target_layer = resolve_target_layer(model, "convnext_tiny")
    tensor = torch.randn(1, 3, 224, 224)
    cam, _ = grad_cam(model, target_layer, tensor)

    image = Image.new("RGB", (64, 48), color=(10, 10, 10))
    overlay = overlay_on_image(image, cam)
    assert overlay.size == image.size  # heatmap resized back to the original image
