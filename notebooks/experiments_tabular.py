import mlflow
import numpy as np
import pandas as pd
import torch.cuda

from experiments.tabular import preprocessing as prp
from experiments.tabular.models import NewsFCNN
from experiments.tabular.distillation import train_baseline
from torch.utils.data import DataLoader
from torch import nn
from sklearn.metrics import accuracy_score


if __name__ == "__main__":

    torch.manual_seed(1234)
    PATH = "../data/OnlineNewsPopularity/OnlineNewsPopularity.csv"
    BATCH_SIZE = 64
    EPOCHS = 10
    # Split the data
    news_df = prp.load_news_df(PATH)
    split_data = prp.NewsSplitPreprocess(news_df, validation=False, train_size=30_000)
    data = split_data()

    x_train, y_train = data[:2]  # Initial split
    x_test, y_test = data[2:]  # Our test data

    further_split = prp.NewsSplitPreprocess(
        pd.DataFrame(np.hstack([x_train, y_train.reshape(-1, 1)])),
        train_size=20_000,
        validation_size=5_000,
        scale=False,
    )
    further_split_data = further_split()
    x_train_old, y_train_old = further_split_data[:2]
    x_train_delta, y_train_delta = further_split_data[2:4]
    x_val, y_val = further_split_data[4:]

    train_old_dataset = prp.NewsDataset(x_train_old, y_train_old)
    train_new_dataset = prp.NewsDataset(
        np.vstack([x_train_old, x_train_delta]), np.hstack([y_train_old, y_train_delta])
    )
    val_dataset = prp.NewsDataset(x_val, y_val)
    test_dataset = prp.NewsDataset(x_test, y_test)
    train_old_dataloader = DataLoader(
        train_old_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    train_new_dataloader = DataLoader(
        train_new_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    # Experiments start
    with mlflow.start_run(run_name="Knowledge distillation"):

        device = "cuda" if torch.cuda.is_available() else "cpu"
        baseline_model = NewsFCNN().to(device)
        teacher_model = NewsFCNN().to(device)

        teacher_loss_fn = nn.CrossEntropyLoss()
        teacher_optimizer = torch.optim.Adam(teacher_model.parameters(), lr=1e-3)

        baseline_loss_fn = nn.CrossEntropyLoss()
        baseline_optimizer = torch.optim.Adam(baseline_model.parameters(), lr=1e-3)

        train_baseline(
            train_old_dataloader,
            val_dataloader,
            teacher_model,
            teacher_optimizer,
            teacher_loss_fn,
            None,
            device,
            EPOCHS,
        )

        train_baseline(
            train_new_dataloader,
            val_dataloader,
            baseline_model,
            baseline_optimizer,
            baseline_loss_fn,
            None,
            device,
            EPOCHS,
        )

        x_test, y_test = torch.Tensor(x_test).to(device), torch.Tensor(y_test).to(
            device
        )
        teacher_model.eval()
        baseline_model.eval()
        teacher_pred = teacher_model(x_test).softmax(1).argmax(1).to("cpu").numpy()
        baseline_pred = baseline_model(x_test)
        baseline_churn = 1.0 - accuracy_score(
            teacher_pred,
            baseline_pred.softmax(1).argmax(1).to("cpu").numpy(),
        )
        mlflow.log_metric(key="baseline_churn", value=baseline_churn)
        mlflow.pytorch.log_model(teacher_model, "teacher_model")

        # Distillation
        alphas = [0.2, 0.4, 0.6, 0.8]

        y_test = y_test.to("cpu").numpy()
        for alpha in alphas:
            student_model = NewsFCNN().to(device)
            student_loss_fn = nn.CrossEntropyLoss()
            student_optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-3)

            train_baseline(
                train_new_dataloader,
                val_dataloader,
                student_model,
                student_optimizer,
                student_loss_fn,
                teacher_model,
                device,
                EPOCHS,
                alpha=alpha,
            )

            student_model.eval()
            student_pred = student_model(x_test).softmax(1).argmax(1).to("cpu").numpy()
            distillation_churn = 1.0 - accuracy_score(
                teacher_pred,
                student_pred,
            )
            good_churn = (
                (y_test != teacher_pred) & (y_test == student_pred)
            ).sum() / len(y_test)
            bad_churn = (
                (y_test == teacher_pred) & (y_test != student_pred)
            ).sum() / len(y_test)
            mlflow.log_metric(key=f"distilled_churn", value=distillation_churn)
            mlflow.log_metric(key="good_churn", value=good_churn)
            mlflow.log_metric(key="bad_churn", value=bad_churn)
            mlflow.log_metric(key="win_loss_ratio", value=good_churn / bad_churn)
            mlflow.log_metric(
                key="churn_ratio", value=distillation_churn / baseline_churn
            )
            mlflow.pytorch.log_model(student_model, f"student_model_{alpha}")

            print(f"alpha={alpha} | churn={distillation_churn}")
