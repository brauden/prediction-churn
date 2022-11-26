from abc import ABC, abstractmethod

import torch
from typing import Optional
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader


class ChurnReduction(ABC):
    def __init__(self, device: str) -> None:
        self.device = device

    @abstractmethod
    def transform(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        ...


class Distillation(ChurnReduction):
    def __init__(self, device: str, lambda_: float = 0.5) -> None:
        super(Distillation, self).__init__(device=device)
        self.lambda_ = lambda_
        self.device = device

    def transform(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        y_distill = self.lambda_ * y_pred.softmax(1) + (1.0 - self.lambda_) * y_true
        y_distill.to(self.device)
        return y_distill


class AnchorRCP(ChurnReduction):
    def __init__(
        self, device: str, alpha: float = 0.5, eps: float = 1.0, classes: int = 2
    ) -> None:
        super(AnchorRCP, self).__init__(device=device)
        self.alpha = alpha
        self.eps = eps
        self.classes = classes

    def transform(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        y_rcp = torch.where(
            y_pred.softmax(1).round() == y_true,
            self.alpha * y_pred.softmax(1) + (1.0 - self.alpha) * y_true,
            self.alpha * y_true
            + self.alpha / self.classes * torch.ones((len(y_true), self.classes)),
        )
        return y_rcp


class Train:
    def __init__(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: Optional[torch.utils.data.DataLoader],
        student_model: nn.Module,
        student_loss: nn.Module,
        student_optimizer: torch.optim.Optimizer,
        teacher_model: Optional[nn.Module] = None,
        churn_transform: Optional[ChurnReduction] = None,
        device: str = "cpu",
        epochs: int = 10,
    ) -> None:
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.student_loss = student_loss
        self.student_optimizer = student_optimizer
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
                x, y = x.to(self.device), y.to(torch.device)

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


def experiment_metrics(
    y_true: torch.Tensor,
    y_teacher: torch.Tensor,
    y_pred: torch.Tensor,
    baseline_churn: float,
):

    y_teacher = y_teacher.softmax(1).argmax(1).to("cpu").numpy()
    y_pred = y_pred.softmax(1).argmax(1).to("cpu").numpy()
    y_true = y_true.softmax(1).argmax(1).to("cpu").numpy()

    churn = 1.0 - (y_teacher == y_pred).sum() / len(y_teacher)
    good_churn = ((y_true != y_teacher) & (y_true == y_pred)).sum() / len(y_true)
    bad_churn = ((y_true == y_teacher) & (y_true != y_pred)).sum() / len(y_true)
    win_loss_ratio = good_churn / bad_churn
    churn_ratio = churn / baseline_churn
    accuracy = (y_true == y_pred).sum() / len(y_teacher)
    metrics = dict(
        churn=churn,
        good_churn=good_churn,
        bad_churn=bad_churn,
        win_loss_ratio=win_loss_ratio,
        churn_ratio=churn_ratio,
        accuracy=accuracy,
    )
    return metrics
