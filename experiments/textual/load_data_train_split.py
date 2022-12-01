import pandas as pd
import numpy as np


def load_data_train_split(data, train: int = 70, test: int = 10):
    """
    Load the dataset
    :param data: dataframe
    :param train: fraction of train dataset
    :param train: fraction of test dataset
    :return: train, validation and test set.
    """
    # imdb_df = pd.read_csv(base_csv)

    train, validate, test = np.split(
        data.sample(frac=1, random_state=42),
        [int(train * 0.01 * len(data)), int(test * 0.01 * len(data))],
    )

    x_train, y_train = np.array(train.iloc[:, 0]), np.array(train.iloc[:, 1])
    x_valid, y_valid = np.array(validate.iloc[:, 0]), np.array(validate.iloc[:, 1])
    x_test, y_test = np.array(test.iloc[:, 0]), np.array(test.iloc[:, 1])

    return x_train, x_valid, x_test, y_train, y_valid, y_test


if __name__ == "__main__":
    load_data_train_split()
