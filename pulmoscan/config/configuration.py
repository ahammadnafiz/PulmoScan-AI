from pathlib import Path

from pulmoscan.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from pulmoscan.entity.config_entity import (
    DataIngestionConfig,
    EvaluationConfig,
    PrepareBaseModelConfig,
    TrainingConfig,
)
from pulmoscan.utils.common import create_directories, read_yaml


class ConfigurationManager:
    """Reads config.yaml + params.yaml and builds typed stage configs."""

    def __init__(
        self,
        config_filepath: Path = CONFIG_FILE_PATH,
        params_filepath: Path = PARAMS_FILE_PATH,
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])
        return DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_URL=config.source_URL,
            local_data_file=Path(config.local_data_file),
            unzip_dir=Path(config.unzip_dir),
        )

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        params = self.params
        create_directories([config.root_dir])
        return PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            backbone=params.BACKBONE,
            pretrained=params.PRETRAINED,
            freeze_backbone=params.FREEZE_BACKBONE,
            num_classes=params.NUM_CLASSES,
            image_size=params.IMAGE_SIZE,
        )

    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        params = self.params
        create_directories([Path(training.root_dir)])
        return TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
            data_root=Path(self.config.data_root),
            class_names_path=Path(training.class_names_path),
            backbone=params.BACKBONE,
            num_classes=params.NUM_CLASSES,
            image_size=params.IMAGE_SIZE,
            epochs=params.EPOCHS,
            batch_size=params.BATCH_SIZE,
            learning_rate=params.LEARNING_RATE,
            weight_decay=params.WEIGHT_DECAY,
            fine_tune_epochs=params.FINE_TUNE_EPOCHS,
            fine_tune_lr=params.FINE_TUNE_LR,
            early_stopping_patience=params.EARLY_STOPPING_PATIENCE,
            label_smoothing=params.LABEL_SMOOTHING,
            use_class_weights=params.USE_CLASS_WEIGHTS,
            num_workers=params.NUM_WORKERS,
            val_split=params.VAL_SPLIT,
            seed=params.SEED,
            augmentation=params.AUGMENTATION,
            k_folds=params.get("K_FOLDS", 5),
            mlflow_tracking_uri=params.get("MLFLOW_TRACKING_URI", "mlruns"),
            mlflow_experiment_name=params.get("MLFLOW_EXPERIMENT_NAME", "PulmoScan"),
        )

    def get_evaluation_config(self) -> EvaluationConfig:
        evaluation = self.config.evaluation
        training = self.config.training
        params = self.params
        create_directories([Path(evaluation.root_dir)])
        return EvaluationConfig(
            trained_model_path=Path(training.trained_model_path),
            data_root=Path(self.config.data_root),
            scores_path=Path(evaluation.scores_path),
            image_size=params.IMAGE_SIZE,
            batch_size=params.BATCH_SIZE,
            num_workers=params.NUM_WORKERS,
            use_tta=params.USE_TTA,
            all_params=dict(params),
            mlflow_tracking_uri=params.get("MLFLOW_TRACKING_URI", "mlruns"),
            mlflow_experiment_name=params.get("MLFLOW_EXPERIMENT_NAME", "PulmoScan"),
        )

    def get_ensemble_evaluation_config(self) -> EvaluationConfig:
        """Evaluation config for the fold ensemble — same settings, but scored
        into ``ensemble_scores_path`` so single and ensemble metrics coexist."""
        evaluation = self.config.evaluation
        training = self.config.training
        params = self.params
        create_directories([Path(evaluation.root_dir)])
        return EvaluationConfig(
            trained_model_path=Path(training.trained_model_path),
            data_root=Path(self.config.data_root),
            scores_path=Path(evaluation.ensemble_scores_path),
            image_size=params.IMAGE_SIZE,
            batch_size=params.BATCH_SIZE,
            num_workers=params.NUM_WORKERS,
            use_tta=params.USE_TTA,
            all_params=dict(params),
            mlflow_tracking_uri=params.get("MLFLOW_TRACKING_URI", "mlruns"),
            mlflow_experiment_name=params.get("MLFLOW_EXPERIMENT_NAME", "PulmoScan"),
        )
