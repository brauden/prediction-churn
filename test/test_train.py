import torch
from torch import nn
import numpy as np
import pytest
from churn import Distillation, ChurnTrain


def load_test_data():
    with open("data/popularity.npy", "rb") as f:
        data = np.load(f)
    return data


data = load_test_data()


class NewsFCNN(nn.Module):
    def __init__(self):
        super(NewsFCNN, self).__init__()
        self.fc_sequence = nn.Sequential(
            nn.Linear(58, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 2)
        )

    def forward(self, x):
        logits = self.fc_sequence(x)
        return logits


class TestChurnTrain:
    @pytest.fixture
    def model_init(self) -> tuple:
        base_model = NewsFCNN()
        base_model.load_state_dict(torch.load("data/teacher.pth"))
        model = NewsFCNN()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train = ChurnTrain(
            model,
            (data[:, :-1], data[:, -1]),
            (data[:, :-1], data[:, -1]),
            nn.CrossEntropyLoss(),
            optimizer,
            3,
            64,
            2,
            Distillation(lambda_=0.5),
            base_model,
            None,
        )
        return model, optimizer, train

    def test_data_load(self):
        assert data.shape == (9644, 59)

    def test_dataloader_creation(self, model_init):
        _, _, train = model_init
        batch = next(iter(train.train_data))
        assert isinstance(train.train_data, torch.utils.data.DataLoader)
        assert isinstance(train.val_data, torch.utils.data.DataLoader)
        assert len(batch) == 2
        assert batch[0].shape == (64, 58)
        assert batch[1].shape == (64, 2)

    def test_train_step(self, model_init):
        model, optimizer, train = model_init
        train_data = train.train_data
        train.model.to(train.device)
        x, y = next(iter(train_data))
        x, y = x.to(torch.float32).to(train.device), y.to(torch.float32).to(
            train.device
        )
        loss = train.train_step(x, y)
        assert isinstance(loss, float)

    def test_train_fit(self, model_init):
        model, optimizer, train = model_init
        history = train.fit()
        assert len(history["train_losses"]) == 3
        assert len(history["val_losses"]) == 3
