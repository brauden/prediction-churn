"""
Preprocessing file for all data sets.
"""
import pandas as pd
import numpy as np
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
    def __init__(
        self, data: pd.DataFrame, seed: int = 1234, train_size: float = 0.7
    ) -> None:
        self.data = data
        self.seed = seed
        self.train_size = train_size
        self.scaler = StandardScaler()

    def _split_preprocess_news(self) -> tuple[np.ndarray, ...]:
        train_df, test_df = train_test_split(
            self.data, random_state=self.seed, train_size=self.train_size
        )
        x_train, y_train = train_df.iloc[:, :-1], train_df.iloc[:, -1]
        x_test, y_test = test_df.iloc[:, :-1], test_df.iloc[:, -1]
        x_train = self.scaler.fit_transform(x_train)
        x_test = self.scaler.transform(x_test)
        return x_train, y_train, x_test, y_test

    def __call__(self) -> tuple[np.ndarray, ...]:
        return self._split_preprocess_news()


class NewsDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx, :], self.y[idx]
