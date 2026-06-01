from pulmoscan import logger
from pulmoscan.components.model_trainer import Training
from pulmoscan.config.configuration import ConfigurationManager

STAGE_NAME = "Training stage"


class ModelTrainingPipeline:
    def main(self) -> None:
        config = ConfigurationManager()
        training_config = config.get_training_config()
        training = Training(config=training_config)
        training.train()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        ModelTrainingPipeline().main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
