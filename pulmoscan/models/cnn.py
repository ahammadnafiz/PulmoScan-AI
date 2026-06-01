"""Model factory and preprocessing transforms.

Single source of truth for the network architecture and the image
preprocessing pipeline. Both the training pipeline and the serving API
import from here, which guarantees train/serve consistency — the model
is built and the images are normalized identically in both paths.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode

# ImageNet statistics — the pretrained backbones expect inputs normalized
# with these per-channel means/stds.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

SUPPORTED_BACKBONES = ("resnet50", "convnext_tiny", "efficientnet_v2_s")


def build_model(
    backbone: str = "resnet50",
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_backbone: bool = True,
) -> nn.Module:
    """Build a transfer-learning CNN with a fresh classifier head.

    Args:
        backbone: One of ``SUPPORTED_BACKBONES``.
        num_classes: Number of output classes.
        pretrained: Load ImageNet-pretrained weights.
        freeze_backbone: Freeze every parameter except the new head.

    Returns:
        A ``torch.nn.Module`` ready for training or inference.
    """
    backbone = backbone.lower()

    if backbone == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        _freeze(model, freeze_backbone)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif backbone == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.convnext_tiny(weights=weights)
        _freeze(model, freeze_backbone)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)

    elif backbone == "efficientnet_v2_s":
        weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_v2_s(weights=weights)
        _freeze(model, freeze_backbone)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(
            f"Unsupported backbone '{backbone}'. Choose one of {SUPPORTED_BACKBONES}."
        )

    return model


def _freeze(model: nn.Module, freeze: bool) -> None:
    """Freeze all parameters when ``freeze`` is True.

    The classifier head is replaced *after* this call, so the new head's
    freshly created parameters keep ``requires_grad=True`` and remain
    trainable even when the backbone is frozen.
    """
    if freeze:
        for param in model.parameters():
            param.requires_grad = False


def get_train_transforms(image_size: int = 224, augmentation: bool = True) -> transforms.Compose:
    """Preprocessing + optional augmentation for the training split."""
    if augmentation:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return get_eval_transforms(image_size)


def get_eval_transforms(image_size: int = 224) -> transforms.Compose:
    """Deterministic preprocessing for validation, evaluation, and serving."""
    return transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size), interpolation=InterpolationMode.BILINEAR
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
