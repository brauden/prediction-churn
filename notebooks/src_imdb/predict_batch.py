from sklearn.preprocessing import OneHotEncoder

from proofpoint.notebooks.src_imdb import load_data_train_split
from src_imdb import build_vocab
from src_imdb import HyperParams
from src_imdb import predict_sentiment
import numpy as np
import torch

def s_prediction_batch(student_model, data):
  x_train, _, x_test, _, _, y_test = load_data_train_split.load_data_train_split(data)

  hparams = HyperParams.HyperParams()
  vocab = build_vocab.build_vocab(x_train,5,hparams)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model =student_model # torch.load(os.path.join(CHECKPOINT_FOLDER,f'{teacher_model}'))
  model.eval()

  one_hot = OneHotEncoder()
  y_true_test = np.where(y_test == "positive", 1, 0)

  pred_lst = [predict_sentiment.predict_sentiment(text, model, vocab, device) for text in x_test]
 
  pred_label = np.array([[1-j, j] if i ==1 else [j, 1-j] for i,j in pred_lst])
 
  return [pred_label, y_true_test]


def t_prediction_batch(teacher_model, old_data, new_data):
  _, _, x_test_new, _, _, y_test_new = load_data_train_split.load_data_train_split(new_data)
  x_train, _, _, _, _, _ = load_data_train_split.load_data_train_split(old_data)

  hparams = HyperParams.HyperParams()
  vocab = build_vocab.build_vocab(x_train,5,hparams)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model =teacher_model # torch.load(os.path.join(CHECKPOINT_FOLDER,f'{teacher_model}'))
  model.eval()

  one_hot = OneHotEncoder()
  y_true_test = np.where(y_test_new == "positive", 1, 0)

  pred_lst = [predict_sentiment.predict_sentiment(text, model, vocab, device) for text in x_test_new]
 
  pred_label = np.array([[1-j, j] if i ==1 else [j, 1-j] for i,j in pred_lst])
 
  return [pred_label,y_true_test]

if __name__ == '__main__':
    s_prediction_batch()
    t_prediction_batch()