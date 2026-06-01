import os

from pulmoscan import logger
from pulmoscan.components.data_ingestion import DataIngestion
from pulmoscan.config.configuration import ConfigurationManager

STAGE_NAME = "Data Ingestion stage"


class DataIngestionTrainingPipeline:
    def main(self) -> None:
        config = ConfigurationManager()
        data_root = config.config.data_root
        if os.path.isdir(os.path.join(data_root, "train")):
            logger.info(f"Dataset already present at '{data_root}/' — skipping download.")
            return
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        DataIngestionTrainingPipeline().main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
