import os
import zipfile

import gdown

from pulmoscan import logger
from pulmoscan.entity.config_entity import DataIngestionConfig
from pulmoscan.utils.common import get_size


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self) -> str:
        """Download the dataset zip from the configured Google Drive URL."""
        try:
            dataset_url = self.config.source_URL
            zip_download_dir = str(self.config.local_data_file)
            os.makedirs(self.config.root_dir, exist_ok=True)
            logger.info(f"Downloading data from {dataset_url} into {zip_download_dir}")

            file_id = dataset_url.split("/")[-2]
            prefix = "https://drive.google.com/uc?/export=download&id="
            gdown.download(prefix + file_id, zip_download_dir)

            logger.info(f"Downloaded data into {zip_download_dir} ({get_size(self.config.local_data_file)})")
            return zip_download_dir
        except Exception as e:
            raise e

    def extract_zip_file(self) -> None:
        """Extract the downloaded zip into the unzip directory."""
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, "r") as zip_ref:
            zip_ref.extractall(unzip_path)
        logger.info(f"Extracted dataset into {unzip_path}")
