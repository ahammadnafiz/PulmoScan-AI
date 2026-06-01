"""Dataset construction shared by the trainer, the evaluator, and the notebook.

Handles the Chest-CT dataset quirk where train/valid class folders carry
TNM-staging suffixes (e.g. ``adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib``)
while test folders do not (``adenocarcinoma``). All folder names are
normalized to a canonical class so every split shares one label space.
"""

from __future__ import annotations

import os
from collections import Counter

import torch
from PIL import Image
from torch.utils.data import Dataset, Subset, random_split
from torchvision.datasets import ImageFolder

from pulmoscan import logger
from pulmoscan.models import get_eval_transforms, get_train_transforms


def canonical_class(folder_name: str) -> str:
    """Map a raw class-folder name to its canonical class.

    ``adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib`` -> ``adenocarcinoma``
    ``large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa`` -> ``large.cell.carcinoma``
    ``normal`` -> ``normal``
    """
    return folder_name.split("_", 1)[0]


def _remap_to_canonical(dataset: ImageFolder) -> ImageFolder:
    """Rewrite an ImageFolder's labels onto canonical, sorted class indices."""
    canonical = sorted({canonical_class(c) for c in dataset.classes})
    canonical_to_idx = {name: i for i, name in enumerate(canonical)}
    old_to_new = {
        old_idx: canonical_to_idx[canonical_class(name)]
        for old_idx, name in enumerate(dataset.classes)
    }
    dataset.samples = [(path, old_to_new[target]) for path, target in dataset.samples]
    dataset.imgs = dataset.samples
    dataset.targets = [old_to_new[t] for t in dataset.targets]
    dataset.classes = canonical
    dataset.class_to_idx = canonical_to_idx
    return dataset


def _imagefolder(directory: str, transform) -> ImageFolder:
    return _remap_to_canonical(ImageFolder(directory, transform=transform))


def build_datasets(
    data_root: str,
    image_size: int,
    augmentation: bool,
    val_split: float,
    seed: int,
):
    """Return ``(train_ds, val_ds, class_names, train_targets)``.

    Uses ``data_root/train`` and ``data_root/valid`` when present, otherwise
    splits a single class-folder directory by ``val_split``. ``train_targets``
    is the list of integer labels for the train split (used for class weights).
    """
    train_transforms = get_train_transforms(image_size, augmentation)
    eval_transforms = get_eval_transforms(image_size)

    train_dir = os.path.join(data_root, "train")
    valid_dir = os.path.join(data_root, "valid")

    if os.path.isdir(train_dir) and os.path.isdir(valid_dir):
        train_ds = _imagefolder(train_dir, train_transforms)
        val_ds = _imagefolder(valid_dir, eval_transforms)
        logger.info(
            f"Explicit splits: {len(train_ds)} train / {len(val_ds)} val | "
            f"classes={train_ds.classes}"
        )
        return train_ds, val_ds, train_ds.classes, list(train_ds.targets)

    # Single directory: deterministic random split. Eval transforms on the
    # validation subset so augmentation never leaks into evaluation.
    full_train = _imagefolder(data_root, train_transforms)
    full_eval = _imagefolder(data_root, eval_transforms)
    class_names = full_train.classes

    n_total = len(full_train)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val
    generator = torch.Generator().manual_seed(seed)
    train_idx, val_idx = random_split(range(n_total), [n_train, n_val], generator=generator)
    train_idx, val_idx = list(train_idx), list(val_idx)

    train_ds = Subset(full_train, train_idx)
    val_ds = Subset(full_eval, val_idx)
    train_targets = [full_train.targets[i] for i in train_idx]
    logger.info(f"Single dir ({n_total} images) -> {n_train} train / {n_val} val")
    return train_ds, val_ds, class_names, train_targets


def build_test_dataset(data_root: str, image_size: int):
    """Return the held-out test ImageFolder (falls back to valid, then root)."""
    eval_transforms = get_eval_transforms(image_size)
    for split in ("test", "valid"):
        directory = os.path.join(data_root, split)
        if os.path.isdir(directory):
            ds = _imagefolder(directory, eval_transforms)
            logger.info(f"Test set: {len(ds)} images from '{split}' | classes={ds.classes}")
            return ds
    return _imagefolder(data_root, eval_transforms)


class PathListDataset(Dataset):
    """Dataset over an explicit ``[(path, label), ...]`` list with a transform.

    Used by k-fold cross-validation so the same pooled image list can be sliced
    into train/val folds while applying train-time augmentation to the training
    fold and deterministic eval transforms to the held-out fold.
    """

    def __init__(self, samples: list[tuple[str, int]], transform) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label


def _pool_samples(data_root: str) -> tuple[list[tuple[str, int]], list[str]]:
    """Pool ``train/`` + ``valid/`` into one canonical-labelled sample list.

    The held-out ``test/`` split is never read here — k-fold validation must
    only ever see train+valid, so the test set stays a clean generalization probe.
    """
    pooled: list[tuple[str, int]] = []
    class_names: list[str] | None = None
    for split in ("train", "valid"):
        directory = os.path.join(data_root, split)
        if not os.path.isdir(directory):
            continue
        ds = _imagefolder(directory, transform=None)
        if class_names is None:
            class_names = ds.classes
        elif ds.classes != class_names:
            raise ValueError(f"Class mismatch between splits: {ds.classes} != {class_names}")
        pooled.extend(ds.samples)
    if class_names is None:
        raise ValueError(f"No train/ or valid/ directory under {data_root!r}.")
    return pooled, class_names


def build_kfold_datasets(
    data_root: str,
    image_size: int,
    augmentation: bool,
    k: int,
    seed: int,
):
    """Yield ``(fold_idx, train_ds, val_ds, class_names, train_targets)`` per fold.

    Stratified by class so every fold preserves the class distribution. Splits
    only the pooled ``train+valid`` data; ``test/`` is untouched, guaranteeing the
    final test-set evaluation stays an unbiased, leakage-free generalization check.
    """
    from sklearn.model_selection import StratifiedKFold

    pooled, class_names = _pool_samples(data_root)
    labels = [label for _, label in pooled]
    train_transforms = get_train_transforms(image_size, augmentation)
    eval_transforms = get_eval_transforms(image_size)

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    logger.info(
        f"K-fold: pooled {len(pooled)} train+valid images into {k} stratified folds "
        f"| classes={class_names}"
    )
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(pooled, labels)):
        train_samples = [pooled[i] for i in train_idx]
        val_samples = [pooled[i] for i in val_idx]
        train_ds = PathListDataset(train_samples, train_transforms)
        val_ds = PathListDataset(val_samples, eval_transforms)
        train_targets = [label for _, label in train_samples]
        yield fold_idx, train_ds, val_ds, class_names, train_targets


def compute_class_weights(targets: list[int], num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights for CrossEntropyLoss, mean-normalized to ~1."""
    counts = Counter(targets)
    total = len(targets)
    weights = torch.tensor(
        [total / (num_classes * max(counts.get(c, 0), 1)) for c in range(num_classes)],
        dtype=torch.float32,
    )
    return weights / weights.mean()
