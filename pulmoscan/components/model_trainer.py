import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from pulmoscan import logger
from pulmoscan.entity.config_entity import TrainingConfig
from pulmoscan.models import build_model
from pulmoscan.utils.common import get_device, set_seed
from pulmoscan.utils.data import build_datasets, build_kfold_datasets, compute_class_weights


class Training:
    """Two-phase transfer learning for best accuracy:

    Phase 1 — train the classifier head with the backbone frozen.
    Phase 2 — unfreeze the backbone and fine-tune end to end at a low LR.

    Class-weighted, label-smoothed cross-entropy handles the dataset's
    imbalance; a cosine LR schedule and best-val-accuracy checkpointing keep
    the result stable.
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = get_device()
        self.best_val_acc = 0.0

    def train(self) -> None:
        """Single-model training on the fixed train/valid split (DVC default path)."""
        set_seed(self.config.seed)
        logger.info(f"Training on device: {self.device}")

        train_ds, val_ds, class_names, train_targets = build_datasets(
            data_root=str(self.config.data_root),
            image_size=self.config.image_size,
            augmentation=self.config.augmentation,
            val_split=self.config.val_split,
            seed=self.config.seed,
        )
        self._check_num_classes(class_names)

        best = self._train_one(
            train_ds,
            val_ds,
            class_names,
            train_targets,
            out_path=Path(self.config.trained_model_path),
            tag="single",
        )

        with open(self.config.class_names_path, "w") as f:
            json.dump({"class_names": class_names}, f, indent=2)
        logger.info(f"Training complete. Best val_acc={best:.4f}")

    def train_kfold(self, k: int, out_dir: Path) -> list[float]:
        """Stratified k-fold training. Trains one model per fold from the same
        base-model init, saving ``model_fold{i}.pt`` into ``out_dir``.

        Only the pooled train+valid data is split into folds — the test set is
        never touched, so the per-fold val accuracies form an unbiased CV
        estimate and the saved checkpoints can be ensembled at serving time.

        Returns the per-fold best validation accuracies.
        """
        set_seed(self.config.seed)
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"K-fold training (k={k}) on device: {self.device}")

        fold_scores: list[float] = []
        class_names: list[str] = []
        for fold_idx, train_ds, val_ds, class_names, train_targets in build_kfold_datasets(
            data_root=str(self.config.data_root),
            image_size=self.config.image_size,
            augmentation=self.config.augmentation,
            k=k,
            seed=self.config.seed,
        ):
            self._check_num_classes(class_names)
            out_path = out_dir / f"model_fold{fold_idx}.pt"
            logger.info(f"───── Fold {fold_idx + 1}/{k} → {out_path.name} ─────")
            best = self._train_one(
                train_ds,
                val_ds,
                class_names,
                train_targets,
                out_path=out_path,
                tag=f"fold{fold_idx}",
            )
            fold_scores.append(best)

        # Write class names beside the fold checkpoints (not to the single-model
        # path) so the k-fold and single-training DVC stages keep disjoint outputs.
        with open(out_dir / "class_names.json", "w") as f:
            json.dump({"class_names": class_names}, f, indent=2)
        mean = sum(fold_scores) / len(fold_scores)
        logger.info(
            f"K-fold complete. Per-fold best val_acc={[round(s, 4) for s in fold_scores]} "
            f"| CV mean={mean:.4f}"
        )
        return fold_scores

    def _check_num_classes(self, class_names: list[str]) -> None:
        if len(class_names) != self.config.num_classes:
            raise ValueError(
                f"NUM_CLASSES={self.config.num_classes} in params.yaml but the dataset "
                f"has {len(class_names)} classes {class_names}. Update params.yaml."
            )

    def _train_one(
        self, train_ds, val_ds, class_names, train_targets, out_path: Path, tag: str
    ) -> float:
        """Train a single model (both phases) from the prepared base-model init,
        checkpointing the best-val-acc weights to ``out_path``. Returns best val_acc."""
        pin = self.device.type == "cuda"
        # persistent_workers keeps DataLoader workers alive across epochs —
        # essential on macOS (spawn) where re-spawning every epoch dominates
        # runtime for a small dataset like this.
        persistent = self.config.num_workers > 0
        train_loader = DataLoader(
            train_ds,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=pin,
            persistent_workers=persistent,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=pin,
            persistent_workers=persistent,
        )

        model = build_model(
            backbone=self.config.backbone,
            num_classes=self.config.num_classes,
            pretrained=False,
            freeze_backbone=False,
        )
        model.load_state_dict(
            torch.load(self.config.updated_base_model_path, map_location="cpu", weights_only=True)
        )
        model.to(self.device)

        weight = None
        if self.config.use_class_weights:
            weight = compute_class_weights(train_targets, self.config.num_classes).to(self.device)
            logger.info(f"[{tag}] class weights: {weight.cpu().tolist()}")
        criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=self.config.label_smoothing)

        # Best-val tracking is per-model so each fold checkpoints independently.
        self.best_val_acc = 0.0

        # ── Phase 1: head only (backbone frozen) ──────────────────────
        self._freeze_backbone(model, freeze=True)
        self._run_phase(
            model,
            train_loader,
            val_loader,
            criterion,
            class_names,
            out_path,
            epochs=self.config.epochs,
            lr=self.config.learning_rate,
            phase=f"{tag}/head",
        )

        # ── Phase 2: fine-tune the whole network ──────────────────────
        if self.config.fine_tune_epochs > 0:
            self._freeze_backbone(model, freeze=False)
            self._run_phase(
                model,
                train_loader,
                val_loader,
                criterion,
                class_names,
                out_path,
                epochs=self.config.fine_tune_epochs,
                lr=self.config.fine_tune_lr,
                phase=f"{tag}/finetune",
            )
        return self.best_val_acc

    def _run_phase(
        self, model, train_loader, val_loader, criterion, class_names, out_path, epochs, lr, phase
    ):
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=self.config.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
        logger.info(f"[{phase}] {sum(p.numel() for p in params):,} trainable params, lr={lr}")

        patience = self.config.early_stopping_patience
        phase_best, stale = 0.0, 0
        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch(
                model, train_loader, criterion, optimizer, phase, epoch, epochs
            )
            val_loss, val_acc = self._evaluate(model, val_loader, criterion)
            scheduler.step()
            logger.info(
                f"[{phase}] epoch {epoch}/{epochs} | train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
            )
            if val_acc >= self.best_val_acc:
                self.best_val_acc = val_acc
                self._save_checkpoint(model, class_names, out_path)
                logger.info(f"  ↑ new best val_acc={val_acc:.4f} — checkpoint saved")

            # Early stopping on this phase's own best (independent of global best).
            if val_acc > phase_best + 1e-4:
                phase_best, stale = val_acc, 0
            else:
                stale += 1
                if patience and stale >= patience:
                    logger.info(
                        f"[{phase}] early stopping after {epoch} epochs (no gain in {patience})."
                    )
                    break

    def _train_epoch(self, model, loader, criterion, optimizer, phase, epoch, epochs) -> float:
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(loader, desc=f"[{phase}] {epoch}/{epochs}", leave=False):
            images, labels = images.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        return running_loss / len(loader.dataset)

    @torch.no_grad()
    def _evaluate(self, model, loader, criterion):
        model.eval()
        running_loss, correct = 0.0, 0
        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = model(images)
            running_loss += criterion(outputs, labels).item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
        n = len(loader.dataset)
        return running_loss / n, correct / n

    @staticmethod
    def _freeze_backbone(model: nn.Module, freeze: bool) -> None:
        """Freeze/unfreeze everything, then always keep the classifier head trainable."""
        for param in model.parameters():
            param.requires_grad = not freeze
        head = getattr(model, "fc", None) or getattr(model, "classifier", None)
        if head is not None:
            for param in head.parameters():
                param.requires_grad = True

    def _save_checkpoint(self, model, class_names, out_path) -> None:
        """Save a self-describing checkpoint the serving API can load alone."""
        torch.save(
            {
                "state_dict": model.state_dict(),
                "backbone": self.config.backbone,
                "num_classes": self.config.num_classes,
                "class_names": class_names,
                "image_size": self.config.image_size,
            },
            out_path,
        )
