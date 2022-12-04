import random
from itertools import product

import mlflow
import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader

from experiments.tabular import preprocessing as prp
from experiments.tabular.models import NewsFCNN
from experiments.tabular.churn import Train, Distillation, AnchorRCP, experiment_metrics


if __name__ == "__main__":
    # Params
    SEED = random.randint(1, 10_000_000)
    PATH = "data/OnlineNewsPopularity/OnlineNewsPopularity.csv"
    BATCH_SIZE = 64
    EPOCHS = 5
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Split the data
    news_df = prp.load_news_df(PATH)
    split_data = prp.NewsSplitPreprocess(
        news_df, seed=SEED, validation=False, train_size=30_000
    )
    data = split_data()

    x_train, y_train = data[:2]  # Initial split
    x_test, y_test = data[2:]  # Our test data
    x_test, y_test = torch.Tensor(x_test).to(DEVICE), torch.Tensor(y_test).to(DEVICE)

    further_split = prp.NewsSplitPreprocess(
        pd.DataFrame(np.hstack([x_train, y_train.reshape(-1, 1)])),
        train_size=20_000,
        validation_size=5_000,
        scale=False,
        seed=SEED,
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

    # Experiments
    with mlflow.start_run(run_name="Baseline"):
        baseline_train = Train(
            train_new_dataloader,
            val_dataloader,
            NewsFCNN(),
            device=DEVICE,
            epochs=EPOCHS,
        )
        teacher_train = Train(
            train_old_dataloader,
            val_dataloader,
            NewsFCNN(),
            device=DEVICE,
            epochs=EPOCHS,
        )
        baseline_train.fit()
        teacher_train.fit()
        baseline_train.model_mode("eval")
        teacher_train.model_mode("eval")

        teacher_pred = teacher_train(x_test)
        baseline_pred = baseline_train(x_test)
        baseline_metrics = experiment_metrics(y_test, teacher_pred, baseline_pred)
        mlflow.log_metrics(baseline_metrics)

    # Knowledge distillation
    with mlflow.start_run(run_name="Knowledge distillation"):
        lambdas = [0.2, 0.4, 0.6, 0.8]
        for lambda_ in lambdas:
            with mlflow.start_run(run_name=f"KD lambda={lambda_}", nested=True):
                mlflow.log_param("lambda", lambda_)
                distillation = Distillation(device=DEVICE, lambda_=lambda_)
                distillation_train = Train(
                    train_new_dataloader,
                    val_dataloader,
                    NewsFCNN(),
                    teacher_model=teacher_train.get_model,
                    churn_transform=distillation,
                    device=DEVICE,
                    epochs=EPOCHS,
                )
                distillation_train.fit()
                distillation_pred = distillation_train(x_test)
                distillation_metrics = experiment_metrics(
                    y_test, teacher_pred, distillation_pred
                )
                distillation_metrics["churn_ratio_distillation"] = (
                    distillation_metrics["src"] / baseline_metrics["src"]
                )
                mlflow.log_metrics(distillation_metrics)

    # Anchor RCP
    with mlflow.start_run(run_name="Anchor RCP"):
        alphas = [0.2, 0.4, 0.6, 0.8]
        epsilons = [1.0, 0.8, 0.6]
        h_params = list(product(alphas, epsilons))

        for alpha, epsilon in h_params:
            with mlflow.start_run(
                run_name=f"ARCP alpha={alpha}, eps={epsilon}", nested=True
            ):
                mlflow.log_params(dict(alpha=alpha, eps=epsilon))
                anchor = AnchorRCP(alpha=alpha, eps=epsilon, device=DEVICE)
                anchor_train = Train(
                    train_new_dataloader,
                    val_dataloader,
                    NewsFCNN(),
                    teacher_model=teacher_train.get_model,
                    churn_transform=anchor,
                    device=DEVICE,
                    epochs=EPOCHS,
                )
                anchor_train.fit()
                anchor_pred = anchor_train(x_test)
                anchor_metrics = experiment_metrics(y_test, teacher_pred, anchor_pred)
                anchor_metrics["churn_ratio_anchor"] = (
                    anchor_metrics["src"] / baseline_metrics["src"]
                )
                mlflow.log_metrics(anchor_metrics)
