from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    backbone: str
    pretrained: bool
    freeze_backbone: bool
    num_classes: int
    image_size: int


@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path
    data_root: Path
    class_names_path: Path
    backbone: str
    num_classes: int
    image_size: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    fine_tune_epochs: int
    fine_tune_lr: float
    early_stopping_patience: int
    label_smoothing: float
    use_class_weights: bool
    num_workers: int
    val_split: float
    seed: int
    augmentation: bool
    k_folds: int
    mlflow_tracking_uri: str
    mlflow_experiment_name: str


@dataclass(frozen=True)
class EvaluationConfig:
    trained_model_path: Path
    data_root: Path
    scores_path: Path
    image_size: int
    batch_size: int
    num_workers: int
    use_tta: bool
    all_params: dict
    mlflow_tracking_uri: str
    mlflow_experiment_name: str
