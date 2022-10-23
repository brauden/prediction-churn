
from sklearn.preprocessing import OneHotEncoder

from src_imdb import load_imdb
from src_imdb import build_vocab
from src_imdb import HyperParams
from src_imdb import 
import numpy as np
import torch

def distillation(teacher_model, alpha):
  x_train_new, _, _, y_train_new, _, _ = load_imdb.load_imdb(35_000)
  x_train, _, _, y_train, _, _ = load_imdb.load_imdb()

  hparams = HyperParams.HyperParams()
  vocab = build_vocab.build_vocab(x_train,5,hparams)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model =teacher_model # torch.load(os.path.join(CHECKPOINT_FOLDER,f'{teacher_model}'))
  model.eval()

  one_hot = OneHotEncoder()
  y_true_trn = np.where(y_train_new == "positive", 1, 0)
  true_label_trn = one_hot.fit_transform(y_true_trn.reshape(-1, 1))
  true_label_trn = true_label_trn.todense()
  true_label_trn = np.asarray(true_label_trn)

  pred_lst = [predict_sentiment.predict_sentiment(text, model, vocab, device) for text in x_train_new]
 
  pred_label = np.array([[1-j, j] if i ==1 else [j, 1-j] for i,j in pred_lst])
  dist = []
  for true_l, pred_l in zip(true_label_trn, pred_label):
    c = (1 - alpha) * true_l + alpha * pred_l
    dist.append(c)
  return [np.array(dist),pred_label]


if __name__ == '__main__':
    distillation()