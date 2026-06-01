"""Run the full training pipeline end to end (non-DVC entry point).

For incremental, cached runs prefer ``dvc repro``. This script simply
executes every stage in order.
"""

from pulmoscan import logger
from pulmoscan.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from pulmoscan.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from pulmoscan.pipeline.stage_03_model_trainer import ModelTrainingPipeline
from pulmoscan.pipeline.stage_04_evaluation import EvaluationPipeline

STAGES = [
    ("Data Ingestion stage", DataIngestionTrainingPipeline),
    ("Prepare base model stage", PrepareBaseModelTrainingPipeline),
    ("Training stage", ModelTrainingPipeline),
    ("Evaluation stage", EvaluationPipeline),
]


def run() -> None:
    for stage_name, pipeline_cls in STAGES:
        try:
            logger.info(f">>>>>> stage {stage_name} started <<<<<<")
            pipeline_cls().main()
            logger.info(f">>>>>> stage {stage_name} completed <<<<<<\n\nx==========x")
        except Exception as e:
            logger.exception(e)
            raise e


if __name__ == "__main__":
    run()
