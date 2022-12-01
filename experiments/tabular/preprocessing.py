"""
Preprocessing file for all data sets.
"""
import pandas as pd
import numpy as np
import torch
from torch.nn.functional import one_hot
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_news_df(path: str) -> pd.DataFrame:
    """
    Online news popularity dataframe loader
    """
    df = pd.read_csv(path)
    df.columns = [col.strip() for col in df.columns]
    df["shares"] = np.where(df.shares >= 1400, 1, 0)
    df.drop(["url", "timedelta"], axis=1, inplace=True)
    return df


class NewsSplitPreprocess:
    """
    Splits the data into train and test sets
    """

    def __init__(
        self,
        data: pd.DataFrame,
        seed: int = 1234,
        train_size: float = 0.7,
        validation: bool = True,
        validation_size: float = 0.3,
        scale: bool = True
    ) -> None:
        self.data = data
        self.seed = seed
        self.train_size = train_size
        self.val_size = validation_size
        self.validation = validation
        self.scale = scale
        self.scaler = StandardScaler()

    def _split_preprocess_news(self) -> tuple[np.ndarray, ...]:
        train_df, test_df = train_test_split(
            self.data, random_state=self.seed, train_size=self.train_size
        )
        x_train, y_train = train_df.iloc[:, :-1], train_df.iloc[:, -1]
        x_test, y_test = test_df.iloc[:, :-1], test_df.iloc[:, -1]
        if self.scale:
            x_train = self.scaler.fit_transform(x_train)
            x_test = self.scaler.transform(x_test)
        if not self.validation:
            return x_train, y_train.to_numpy(), x_test, y_test.to_numpy()
        else:
            x_train, x_val, y_train, y_val = train_test_split(
                x_train, y_train, random_state=self.seed, test_size=self.val_size
            )
            return (
                x_train.to_numpy() if not self.scale else x_train,
                y_train.to_numpy(),
                x_val.to_numpy() if not self.scale else x_val,
                y_val.to_numpy(),
                x_test.to_numpy() if not self.scale else x_test,
                y_test.to_numpy(),
            )

    def __call__(self) -> tuple[np.ndarray, ...]:
        return self._split_preprocess_news()


class NewsDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = torch.Tensor(y).to(torch.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        labels = one_hot(self.y[idx], num_classes=2)
        return self.X[idx, :], labels
