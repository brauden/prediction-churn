"""
Knowledge distillation procedure
"""
import torch
from typing import Optional
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader


def train_baseline(
    train_dloader: DataLoader,
    val_dloader: DataLoader,
    student_model: nn.Module,
    student_optim,
    student_loss_fn,
    teacher_model: Optional[nn.Module],
    device,
    epochs: int = 10,
    alpha: float = 0.5,

) -> None:
    """
    Training procedure for a teacher model.
    :param alpha:
    :param teacher_model:
    :param train_dloader:
    :param val_dloader:
    :param student_model:
    :param student_optim:
    :param student_loss_fn:
    :param epochs:
    :param device:
    """
    size = len(train_dloader.dataset)

    if teacher_model:
        teacher_model.eval()

    for epoch in range(1, epochs + 1):
        student_model.train()
        for batch, (X, y) in tqdm(enumerate(train_dloader)):
            X, y = X.to(torch.float), y.to(torch.float)
            X, y = X.to(device), y.to(device)
            y_distill = None

            if teacher_model:
                y_teacher = teacher_model(X)
                y_distill = alpha * y_teacher.softmax(1) + (1. - alpha) * y
                y_distill.to(device)

            if y_distill is not None:
                y_use = y_distill
            else:
                y_use = y

            # Compute prediction error
            pred = student_model(X)
            loss = student_loss_fn(pred, y_use)

            # Backpropagation
            student_optim.zero_grad()
            loss.backward()
            student_optim.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"epoch: {epoch} loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        if val_dloader:
            val_size = len(val_dloader.dataset)
            num_batches = len(val_dloader)
            student_model.eval()
            test_loss, correct = 0, 0
            with torch.no_grad():
                for X, y in val_dloader:
                    X, y = X.to(torch.float), y.to(torch.float)
                    X, y = X.to(device), y.to(device)
                    pred = student_model(X)
                    test_loss += student_loss_fn(pred, y).item()
                    correct += (
                        (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
                    )
            test_loss /= num_batches
            correct /= val_size
            print(
                f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
            )
