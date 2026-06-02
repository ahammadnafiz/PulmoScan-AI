"""Grad-CAM explainability for the served classifier.

Grad-CAM needs gradients, so it cannot run under the ``@torch.no_grad`` ensemble
path used by ``InferenceService.predict``. It runs on a single model: one forward
+ one backward pass, weighting the final feature map's activations by the
gradient of the predicted-class score. The result is a coarse,
class-discriminative heatmap, resized and blended over the original image — what
a practitioner looks at to sanity-check *where* the model looked.
"""

from __future__ import annotations

import base64
import io

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


def resolve_target_layer(model: nn.Module, backbone: str) -> nn.Module:
    """The last spatial feature map to attribute over, per backbone.

    Grad-CAM hooks the deepest layer that still has spatial extent (H×W) before
    global pooling collapses it — that is where class-discriminative localization
    is sharpest.
    """
    backbone = backbone.lower()
    if backbone == "resnet50":
        return model.layer4
    if backbone in ("convnext_tiny", "efficientnet_v2_s"):
        return model.features
    raise ValueError(f"Grad-CAM target layer unknown for backbone '{backbone}'.")


def grad_cam(
    model: nn.Module, target_layer: nn.Module, tensor: torch.Tensor
) -> tuple[np.ndarray, np.ndarray]:
    """Run Grad-CAM for the model's top class on a single input tensor.

    Returns ``(cam, probs)`` where ``cam`` is a ``[H, W]`` map normalized to 0..1
    and ``probs`` is the softmax probability vector.
    """
    activations: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor] = {}

    def _fwd(_m: nn.Module, _inp: object, out: torch.Tensor) -> None:
        activations["value"] = out

    def _bwd(_m: nn.Module, _gin: object, gout: tuple[torch.Tensor, ...]) -> None:
        gradients["value"] = gout[0]

    # Require grad on the input so the backward graph reaches the target layer
    # even if the backbone is frozen (and to silence the no-input-grad warning).
    tensor = tensor.detach().requires_grad_(True)

    fwd_handle = target_layer.register_forward_hook(_fwd)
    bwd_handle = target_layer.register_full_backward_hook(_bwd)
    try:
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]
        class_idx = int(probs.argmax().item())
        model.zero_grad(set_to_none=True)
        logits[0, class_idx].backward()
    finally:
        fwd_handle.remove()
        bwd_handle.remove()

    acts = activations["value"][0]  # [C, H, W]
    grads = gradients["value"][0]  # [C, H, W]
    weights = grads.mean(dim=(1, 2))  # [C] — gradient global-average-pool
    cam = F.relu((weights[:, None, None] * acts).sum(dim=0))  # [H, W]

    cam_np = cam.detach().cpu().numpy()
    cam_np -= cam_np.min()
    peak = cam_np.max()
    if peak > 0:
        cam_np /= peak
    return cam_np, probs.detach().cpu().numpy()


def _jet(cam: np.ndarray) -> np.ndarray:
    """Map a 0..1 array to an RGB ``uint8`` heatmap (jet colormap approximation)."""
    r = np.clip(1.5 - np.abs(4.0 * cam - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * cam - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * cam - 1.0), 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)


def overlay_on_image(image: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Blend a Grad-CAM map (any resolution) over the original image."""
    cam_img = Image.fromarray((cam * 255.0).astype(np.uint8)).resize(
        image.size, Image.BILINEAR
    )
    cam_resized = np.asarray(cam_img, dtype=np.float32) / 255.0
    heat = Image.fromarray(_jet(cam_resized))
    return Image.blend(image.convert("RGB"), heat, alpha)


def to_base64_png(image: Image.Image) -> str:
    """Encode a PIL image as a base64 PNG string (no data-URI prefix)."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()
