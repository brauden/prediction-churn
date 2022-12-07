from typing import Optional, Union

import torch
import tqdm
from numpy import ndarray
from torch import Tensor
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import T_co

from churn.transform import ChurnTransform


TrainData = Union[tuple[ndarray, ndarray], tuple[Tensor, Tensor], DataLoader]


class ChurnTrain:
    def __init__(
        self,
        model: nn.Module,
        train_data: TrainData,
        val_data: Optional[TrainData],
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        epochs: int,
        batch_size: int,
        n_classes: int,
        transform: Optional[ChurnTransform],
        base_model: Optional[nn.Module],
        metrics,  # TODO: add churn related metrics
        shuffle_train: bool = True,
        shuffle_valid: bool = False,
    ) -> None:
        self.model = model
        self.train_data = (
            train_data
            if isinstance(train_data, DataLoader)
            else self._create_dataloader(
                train_data, batch_size, shuffle_train, n_classes
            )
        )
        if val_data is not None:
            self.val_data = (
                val_data
                if isinstance(val_data, DataLoader)
                else self._create_dataloader(
                    val_data, batch_size, shuffle_valid, n_classes
                )
            )
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.transform = transform
        self.base_model = base_model.eval() if base_model is not None else None
        self.metrics = metrics
        self.n_classes = n_classes
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.history = {}

    def _create_dataloader(
        self,
        train_data: tuple,
        batch_size: int,
        shuffle: bool,
        n_classes: int,
    ) -> DataLoader:
        class _DataSet(Dataset):
            def __init__(self, data: tuple, classes: int):
                self.x = torch.tensor(data[0], dtype=torch.float32)
                self.y = torch.tensor(data[1], dtype=torch.int64)
                self.classes = classes

            def __len__(self):
                return len(self.x)

            def __getitem__(self, index) -> T_co:
                return self.x[index], torch.nn.functional.one_hot(
                    self.y[index], num_classes=self.classes
                )

        dset = _DataSet(train_data, n_classes)
        return DataLoader(dset, batch_size=batch_size, shuffle=shuffle)

    def train_step(self, x_train: Tensor, y_train: Tensor) -> float:
        self.model.train()
        preds = self.model(x_train)
        loss = self.loss_fn(preds, y_train)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train_epochs(self, train_losses: list, val_losses: list) -> None:
        for epoch in range(1, self.epochs + 1):
            tmp_losses = []
            for x, y in tqdm.tqdm(self.train_data):
                x, y = x.to(torch.float32).to(self.device), y.to(torch.float32).to(
                    self.device
                )

                y_transformed = None
                if (self.transform is not None) and (self.base_model is not None):
                    y_base_model = self.base_model(x)
                    y_transformed = self.transform.transform(y, y_base_model)

                y_to_use = y if y_transformed is None else y_transformed

                tmp_losses.append(self.train_step(x, y_to_use))
            train_losses.append(sum(tmp_losses) / len(tmp_losses))

            if self.val_data is not None:
                self.validate(val_losses)

    def validate(self, val_losses: list) -> None:
        num_batches = len(self.val_data)
        val_loss = 0.0
        with torch.no_grad():
            for x, y in tqdm.tqdm(self.val_data):
                x, y = x.to(torch.float32).to(self.device), y.to(torch.float32).to(
                    self.device
                )
                val_pred = self.model(x)
                val_loss += self.loss_fn(val_pred, y).item()
            val_loss /= num_batches
            val_losses.append(val_loss)

    def fit(self) -> dict:
        train_losses = []
        val_losses = []

        self.model.to(self.device)

        if self.base_model is not None:
            self.base_model.to(self.device)

        self.train_epochs(train_losses, val_losses)

        self.history["train_losses"] = train_losses
        self.history["val_losses"] = val_losses
        return self.history
