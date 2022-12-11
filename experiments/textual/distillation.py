from sklearn.preprocessing import OneHotEncoder

import load_data_train_split
import build_vocab
import predict_sentiment
import train_val
import numpy as np
import os
import torch


def distillation(teacher_model, old_data, new_data, alpha):
    x_train_new, _, _, y_train_new, _, _ = load_data_train_split.load_data_train_split(
        new_data
    )
    x_train, _, _, y_train, _, _ = load_data_train_split.load_data_train_split(old_data)
    hparams = HyperParams.HyperParams()

    print("Creating Vocabulary for Old Dataset...")
    vocab = build_vocab.build_vocab(x_train, 5, hparams)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = (
        teacher_model  # torch.load(os.path.join(CHECKPOINT_FOLDER,f'{teacher_model}'))
    )
    model.eval()

    one_hot = OneHotEncoder()
    y_true_trn = np.where(y_train_new == "positive", 1, 0)
    true_label_trn = one_hot.fit_transform(y_true_trn.reshape(-1, 1))
    true_label_trn = true_label_trn.todense()
    true_label_trn = np.asarray(true_label_trn)

    print("Predcting Labels with Teacher's Model...")
    pred_lst = [
        predict_sentiment.predict_sentiment(text, model, vocab, device)
        for text in x_train_new
    ]

    print("Distillation of Teacher's Label...")
    pred_label = np.array([[1 - j, j] if i == 1 else [j, 1 - j] for i, j in pred_lst])
    dist = []
    for true_l, pred_l in zip(true_label_trn, pred_label):
        c = (1 - alpha) * true_l + alpha * pred_l
        dist.append(c)

    print("Training Student...")
    org_hyperparams = HyperParams.HyperParams()
    org_hyperparams.OPTIM = "rmsprop"
    org_hyperparams.LR = 0.001
    org_hyperparams.BIDIRECTIONAL = True

    train_val.train_and_test_model_with_hparams(
        new_data, org_hyperparams, f"lstm_student_alpha_{alpha}", distil=np.array(dist)
    )

    CHECKPOINT_FOLDER = "./saved_model"
    path = os.path.join(CHECKPOINT_FOLDER, f"lstm_student_alpha_{alpha}.pth")
    print(f"Best Student Model is Avilable in {path}")


if __name__ == "__main__":
    distillation()
