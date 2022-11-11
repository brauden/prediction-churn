import os
import tarfile
import zipfile

import requests
from requests.exceptions import RequestException


def download_data(url: str, path: str) -> str:
    try:
        name = url.split("/")[-1]
        if not os.path.exists(os.path.join(path, name)):
            print(f"Downloading file {name}")
            response = requests.get(url)
            with open(os.path.join(path, name), "wb") as f:
                f.write(response.content)
                print(f"File: {name} has been saved")
        else:
            print(f"File {name} already exists!")
        return name
    except RequestException as e:
        print(str(e))


def exctract_data(file: str, file_format: str, save_dir: str) -> None:
    if file_format == "gz":
        with tarfile.open(file) as f:
            f.extractall(save_dir)
    elif file_format == "zip":
        with zipfile.ZipFile(file) as f:
            f.extractall(save_dir)
    else:
        print("Unknonw file format")
    print(f"{file} has been exctracted!")
