import torch

from pulmoscan import logger
from pulmoscan.entity.config_entity import PrepareBaseModelConfig
from pulmoscan.models import build_model


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def get_base_model(self) -> None:
        """Build the pretrained backbone with a fresh head and save it."""
        model = build_model(
            backbone=self.config.backbone,
            num_classes=self.config.num_classes,
            pretrained=self.config.pretrained,
            freeze_backbone=False,
        )
        self._save(self.config.base_model_path, model)
        logger.info(f"Base model ({self.config.backbone}) saved to {self.config.base_model_path}")

    def update_base_model(self) -> None:
        """Build the transfer-learning model with the backbone frozen."""
        model = build_model(
            backbone=self.config.backbone,
            num_classes=self.config.num_classes,
            pretrained=self.config.pretrained,
            freeze_backbone=self.config.freeze_backbone,
        )
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(
            f"Updated base model: {trainable:,} trainable / {total:,} total params "
            f"(freeze_backbone={self.config.freeze_backbone})"
        )
        self._save(self.config.updated_base_model_path, model)

    @staticmethod
    def _save(path, model) -> None:
        torch.save(model.state_dict(), path)
