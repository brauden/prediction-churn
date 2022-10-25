"""
Knowledge distillation procedure
"""
import torch
from tqdm import tqdm
from torch import nn


def train_baseline(train_dloader, val_dloader, model, optim, loss_fn, epochs, device):
    size = len(train_dloader.dataset)
    for epoch in range(1, epochs + 1):
        model.train()
        for batch, (X, y) in tqdm(enumerate(train_dloader)):
            X, y = X.to(torch.float), y.to(torch.float)
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optim.zero_grad()
            loss.backward()
            optim.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"epoch: {epoch} loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        val_size = len(val_dloader.dataset)
        num_batches = len(val_dloader)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in val_dloader:
                X, y = X.to(torch.float), y.to(torch.float)
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= val_size
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def distillation_train(
    train_dloader,
    val_dloader,
    epochs,
    teacher_model,
    student_model,
    alpha,
    student_loss,
):
    pass
