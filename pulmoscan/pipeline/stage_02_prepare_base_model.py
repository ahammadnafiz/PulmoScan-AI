from pulmoscan import logger
from pulmoscan.components.prepare_base_model import PrepareBaseModel
from pulmoscan.config.configuration import ConfigurationManager

STAGE_NAME = "Prepare base model stage"


class PrepareBaseModelTrainingPipeline:
    def main(self) -> None:
        config = ConfigurationManager()
        prepare_base_model_config = config.get_prepare_base_model_config()
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        prepare_base_model.get_base_model()
        prepare_base_model.update_base_model()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        PrepareBaseModelTrainingPipeline().main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
