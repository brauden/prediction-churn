"""
Knowledge distillation and anchor RCP labels transformations
for prediction churn reduction.
"""
from abc import ABC, abstractmethod

import torch
from typing import Optional
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader


class ChurnReduction(ABC):
    """
    Abstract class for labels transformation
    """
    def __init__(self, device: str) -> None:
        self.device = device

    @abstractmethod
    def transform(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Returns transformed labels based on the particular implementation
        :param y_true: one-hot encoded torch Tensor
        :param y_pred: probabilty tensor
        :return: transformed torch tensor
        """
        ...


class Distillation(ChurnReduction):
    def __init__(self, device: str, lambda_: float = 0.5) -> None:
        """
        :param device: cpu or cuda. Torch based implementation
        :param lambda_: knowledge distillation hyperparam. May force
        student model's predictions to mimic teacher's model predictions or
        stick to the true labels.
        """
        super(Distillation, self).__init__(device=device)
        self.lambda_ = lambda_

    def transform(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        y_distill = self.lambda_ * y_pred.softmax(1) + (1.0 - self.lambda_) * y_true
        y_distill.to(self.device)
        return y_distill


class AnchorRCP(ChurnReduction):
    def __init__(
        self, device: str, alpha: float = 0.5, eps: float = 1.0, classes: int = 2
    ) -> None:
        """
        :param device: cpu or cuda. Torch based implementation
        :param alpha: similar to lambda param in knowledge distillation. Forces
        new model to mimic the original one when predicted class is correct.
        :param eps: When predicted class is not the same as teacher's model predictions
        RCP multiplies the true tensor by eps.
        :param classes: num of classes in classification task
        """
        super(AnchorRCP, self).__init__(device=device)
        self.alpha = alpha
        self.eps = eps
        self.classes = classes

    def transform(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        y_rcp = torch.where(
            y_pred.softmax(1).round() == y_true,
            self.alpha * y_pred.softmax(1) + (1.0 - self.alpha) * y_true,
            self.eps
            * (
                (1. - self.alpha) * y_true
                + self.alpha
                / self.classes
                * torch.ones((len(y_true), self.classes)).to(self.device)
            ),
        )
        return y_rcp


class Train:
    """
    Class for training models for tabular experiment.
    """
    def __init__(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: Optional[torch.utils.data.DataLoader],
        student_model: nn.Module,
        student_loss: nn.Module = nn.CrossEntropyLoss(),
        student_optimizer: torch.optim.Optimizer = torch.optim.Adam,
        teacher_model: Optional[nn.Module] = None,
        churn_transform: Optional[ChurnReduction] = None,
        device: str = "cpu",
        epochs: int = 10,
        lr: float = 1e-3,
    ) -> None:
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.student_loss = student_loss
        self.student_optimizer = student_optimizer(
            self.student_model.parameters(), lr=lr
        )
        self.churn_transform = churn_transform
        self.device = device
        self.epochs = epochs

    def fit(self):
        self.student_model.to(self.device)

        if self.teacher_model is not None:
            self.teacher_model.eval()

        for epoch in range(1, self.epochs + 1):
            self.student_model.train()

            for batch, (x, y) in tqdm(enumerate(self.train_dataloader)):
                x, y = x.to(torch.float), y.to(torch.float)
                x, y = x.to(self.device), y.to(self.device)

                y_transformed = None
                if (self.teacher_model is not None) and (
                    self.churn_transform is not None
                ):
                    y_teacher_pred = self.teacher_model(x)
                    y_transformed = self.churn_transform.transform(y, y_teacher_pred)
                    y_transformed.to(self.device)

                y_use = y if y_transformed is None else y_transformed
                pred = self.student_model(x)
                loss = self.student_loss(pred, y_use)

                self.student_optimizer.zero_grad()
                loss.backward()
                self.student_optimizer.step()

                if not batch % 100:
                    loss, current = loss.item(), batch * len(x)
                    print(f"epoch: {epoch} loss: {loss:>7f}")

            if self.val_dataloader is not None:
                val_size = len(self.val_dataloader.dataset)
                num_batches = len(self.val_dataloader)
                self.student_model.eval()
                test_loss, correct = 0, 0
                with torch.no_grad():
                    for x, y in self.val_dataloader:
                        x, y = x.to(torch.float), y.to(torch.float)
                        x, y = x.to(self.device), y.to(self.device)
                        pred = self.student_model(x)
                        test_loss += self.student_loss(pred, y).item()
                        correct += (
                            (pred.argmax(1) == y.argmax(1))
                            .type(torch.float)
                            .sum()
                            .item()
                        )
                test_loss /= num_batches
                correct /= val_size
                print(
                    f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
                )

    def model_mode(self, mode: str = "eval") -> None:
        if mode == "eval":
            self.student_model.eval()
        elif mode == "train":
            self.student_model.train()
        else:
            raise ValueError("mode param has to be in {eval, train}")

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        self.model_mode("eval")
        return self.student_model(inputs)

    @property
    def get_model(self):
        return self.student_model


def experiment_metrics(
    y_true: torch.Tensor, y_teacher: torch.Tensor, y_pred: torch.Tensor
) -> dict[str, float]:
    """
    Metrics that we track during our experiments:
    1. Churn
    2. Good churn
    3. Bad churn
    4. Win loss ratio
    5. Accuracy

    Churn ratio needs to be calculated separately.
    :param y_true:
    :param y_teacher:
    :param y_pred:
    :return:
    """

    y_teacher = y_teacher.softmax(1).argmax(1).to("cpu").numpy()
    y_pred = y_pred.softmax(1).argmax(1).to("cpu").numpy()
    y_true = y_true.to("cpu").numpy()

    churn = 1.0 - (y_teacher == y_pred).sum() / len(y_teacher)
    good_churn = ((y_true != y_teacher) & (y_true == y_pred)).sum() / len(y_true)
    bad_churn = ((y_true == y_teacher) & (y_true != y_pred)).sum() / len(y_true)
    win_loss_ratio = good_churn / bad_churn
    accuracy = (y_true == y_pred).sum() / len(y_teacher)
    metrics = dict(
        churn=churn,
        good_churn=good_churn,
        bad_churn=bad_churn,
        win_loss_ratio=win_loss_ratio,
        accuracy=accuracy,
    )
    return metrics
