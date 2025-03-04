import os
import json
from pathlib import Path
from kaggle_crawler import download_notebooks, crawl_descriptions

def ensure_directory_exists(path):
    """Ensure the directory exists, create it if it does not."""
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == "__main__":
    competition = "data-science-bowl-2018"
    local_data_path = "/data/userdata/v-xhong/researcher/RD-Agent/scripts/exp/researcher/training_set/raw_jsons/data-science-bowl-2018"

    # Ensure the competition directory exists
    ensure_directory_exists(local_data_path)

    # Crawl competition descriptions
    des = crawl_descriptions(competition=competition, local_data_path=local_data_path)

    # Download notebooks
    download_notebooks(competition=competition, local_path=local_data_path, num=10)