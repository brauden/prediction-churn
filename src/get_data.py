import os
from utils import download_data, exctract_data


if __name__ == "__main__":
    PATH = "data"  # Assuming that the script is running from the root directory
    urls = [
        "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00332/OnlineNewsPopularity.zip"
    ]

    for url in urls:
        file_name = download_data(url, PATH)
        file_format = url.split(".")[-1]
        exctract_data(os.path.join(PATH, file_name), file_format, PATH)
