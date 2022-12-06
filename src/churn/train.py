from typing import Optional, Union

import torch
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
            else self._create_dataloader(train_data, batch_size, shuffle_train)
        )
        if val_data is not None:
            self.val_data = (
                val_data
                if isinstance(val_data, DataLoader)
                else self._create_dataloader(val_data, batch_size, shuffle_valid)
            )
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.transform = transform
        self.base_model = base_model.eval() if base_model is not None else None
        self.metrics = metrics
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.history = {}

    def _create_dataloader(
        self,
        train_data: tuple,
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader:
        class _DataSet(Dataset):
            def __init__(self, data: tuple):
                self.x = data[0]
                self.y = data[1]

            def __getitem__(self, index) -> T_co:
                return self.x[index], self.y[index]

        dset = _DataSet(train_data)
        return DataLoader(dset, batch_size=batch_size, shuffle=shuffle)

    def train_step(self, x_train: Tensor, y_train: Tensor) -> float:
        self.model.train()
        preds = self.model(x_train)
        loss = self.loss_fn(preds, y_train)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def fit(self) -> dict:
        train_losses = []
        val_losses = []

        self.model.to(self.device)

        if self.base_model is not None:
            self.base_model.to(self.device)

        for epoch in range(self.epochs):
            tmp_losses = []
            for x, y in self.train_data:
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
                num_batches = len(self.val_data)
                val_loss = 0.0
                with torch.no_grad():
                    for x, y in self.val_data:
                        x, y = x.to(torch.float32).to(self.device), y.to(
                            torch.float32
                        ).to(self.device)
                        val_pred = self.model(x)
                        val_loss += self.loss_fn(val_pred, y).item()
                    val_loss /= num_batches
                    val_losses.append(val_loss)

        self.history["train_losses"] = train_losses
        self.history["val_losses"] = val_losses
        return self.history
