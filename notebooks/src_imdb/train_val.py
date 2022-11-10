import tqdm
import torch
import torch.optim as optim
import sys
import random
import functools
import numpy as np
import torch.nn as nn
import os
from sklearn.preprocessing import OneHotEncoder

from src_imdb import load_data_train_split
from src_imdb import build_vocab
from src_imdb import Text_Tokenize
from src_imdb import LSTM


def get_accuracy(prediction, label):
    batch_size, _ = prediction.shape
    predicted_classes = prediction.argmax(dim=-1)
    label_classes = label.argmax(dim=-1)
    correct_predictions = predicted_classes.eq(label_classes).sum() # predicted_classes.eq(label).sum() #########
    accuracy = correct_predictions / batch_size
    return accuracy

def collate(batch, pad_index):
    batch_ids = [torch.LongTensor(i['ids']) for i in batch]
    batch_ids = nn.utils.rnn.pad_sequence(batch_ids, padding_value=pad_index, batch_first=True)
    batch_length = torch.Tensor([i['length'] for i in batch])
    batch_label = torch.Tensor([i['label'] for i in batch])
    batch = {'ids': batch_ids, 'length': batch_length, 'label': batch_label}
    return batch

collate_fn = collate


def train(dataloader, model, criterion, optimizer, scheduler, device):
    model.train()
    epoch_losses = []
    epoch_accs = []

    for batch in tqdm.tqdm(dataloader, desc='training...', file=sys.stdout):
        ids = batch['ids'].to(device)
        length = batch['length']
        label = batch['label'].to(device)
        prediction = model(ids, length)
        loss = criterion(prediction, label)
        accuracy = get_accuracy(prediction, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
        epoch_accs.append(accuracy.item())
        scheduler.step()

    return epoch_losses, epoch_accs

def evaluate(dataloader, model, criterion, device):
    model.eval()
    epoch_losses = []
    epoch_accs = []

    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc='evaluating...', file=sys.stdout):
            ids = batch['ids'].to(device)
            length = batch['length']
            label = batch['label'].to(device)
            # label =torch.argmax(label, dim=1)
            prediction = model(ids, length)
            loss = criterion(prediction, label)
            accuracy = get_accuracy(prediction, label)
            epoch_losses.append(loss.item())
            epoch_accs.append(accuracy.item())

    return epoch_losses, epoch_accs




class ConstantWithWarmup(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        num_warmup_steps: int,
    ):
        self.num_warmup_steps = num_warmup_steps
        super().__init__(optimizer)

    def get_lr(self):
        if self._step_count <= self.num_warmup_steps:
            # warmup
            scale = 1.0 - (self.num_warmup_steps - self._step_count) / self.num_warmup_steps
            lr = [base_lr * scale for base_lr in self.base_lrs]
            self.last_lr = lr
        else:
            lr = self.base_lrs
        return lr

# the folder where the trained model is saved
CHECKPOINT_FOLDER = "./saved_model"

def train_and_test_model_with_hparams(data, hparams, model_type="lstm",distil = None, **kwargs): #

    torch.manual_seed(hparams.SEED)
    random.seed(hparams.SEED)
    # np.random.seed(hparams.SEED)

    x_train, x_valid, x_test, y_train, y_valid, y_test = load_data_train_split.load_data_train_split(data)

    one_hot = OneHotEncoder()

    y_true_trn = np.where(y_train == "positive", 1, 0)
    y_true_val = np.where(y_valid == "positive", 1, 0)
    y_true_test = np.where(y_test == "positive", 1, 0)

    true_label_trn = one_hot.fit_transform(y_true_trn.reshape(-1, 1))
    true_label_trn = true_label_trn.todense()
    true_label_trn = np.asarray(true_label_trn)

    true_label_val = one_hot.fit_transform(y_true_val.reshape(-1, 1))
    true_label_val = true_label_val.todense()
    true_label_val = np.asarray(true_label_val)

    true_label_test = one_hot.fit_transform(y_true_test.reshape(-1, 1))
    true_label_test = true_label_test.todense()
    true_label_test = np.asarray(true_label_test)


    if distil is None:
      # y_dist = np.array(distilled[5]) ###########################################################
      y_train = true_label_trn
    else:
      y_train = distil
    # y_train = true_label_trn
    y_valid = true_label_val
    y_test = true_label_test
    vocab = build_vocab.build_vocab(x_train, hparams=hparams)
    vocab_size = len(vocab)

    train_data = Text_Tokenize.Text_Tokenize(x_train, y_train, vocab, hparams.MAX_LENGTH)
    valid_data = Text_Tokenize.Text_Tokenize(x_valid, y_valid, vocab, hparams.MAX_LENGTH)
    test_data = Text_Tokenize.Text_Tokenize(x_test, y_test, vocab, hparams.MAX_LENGTH)

    collate = functools.partial(collate_fn, pad_index=hparams.PAD_INDEX)

    train_dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=hparams.BATCH_SIZE, collate_fn=collate, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(
        valid_data, batch_size=hparams.BATCH_SIZE, collate_fn=collate)
    test_dataloader = torch.utils.data.DataLoader(
        test_data, batch_size=hparams.BATCH_SIZE, collate_fn=collate)
    
    # Model
    if "override_models_with_gru" in kwargs and kwargs["override_models_with_gru"]:
        pass 
        # model = GRU(
        #     vocab_size, 
        #     hparams.EMBEDDING_DIM, 
        #     hparams.HIDDEN_DIM, 
        #     hparams.OUTPUT_DIM,
        #     hparams.N_LAYERS,
        #     hparams.DROPOUT_RATE, 
        #     hparams.PAD_INDEX,
        #     hparams.BIDIRECTIONAL,
        #     **kwargs)
    else:
        model = LSTM.LSTM(
            vocab_size, 
            hparams.EMBEDDING_DIM, 
            hparams.HIDDEN_DIM, 
            hparams.OUTPUT_DIM,
            hparams.N_LAYERS,
            hparams.DROPOUT_RATE, 
            hparams.PAD_INDEX,
            hparams.BIDIRECTIONAL,
            **kwargs)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Optimization. Lab 2 (a)(b) should choose one of them.
    # DO NOT TOUCH optimizer-specific hyperparameters! (e.g., eps, momentum)
    # DO NOT change optimizer implementations!
    if hparams.OPTIM == "sgd":
        optimizer = optim.SGD(
            model.parameters(), lr=hparams.LR, weight_decay=hparams.WD, momentum=.9)        
    elif hparams.OPTIM == "adagrad":
        optimizer = optim.Adagrad(
            model.parameters(), lr=hparams.LR, weight_decay=hparams.WD, eps=1e-6)
    elif hparams.OPTIM == "adam":
        optimizer = optim.Adam(
            model.parameters(), lr=hparams.LR, weight_decay=hparams.WD, eps=1e-6)
    elif hparams.OPTIM == "rmsprop":
        optimizer = optim.RMSprop(
            model.parameters(), lr=hparams.LR, weight_decay=hparams.WD, eps=1e-6, momentum=.9)
    else:
        raise NotImplementedError("Optimizer not implemented!")

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    # Start training
    best_valid_loss = float('inf')
    train_losses = []
    train_accs = []
    valid_losses = []
    valid_accs = []
    
    # Warmup Scheduler. DO NOT TOUCH!
    WARMUP_STEPS = 200
    lr_scheduler = ConstantWithWarmup(optimizer, WARMUP_STEPS)

    for epoch in range(hparams.N_EPOCHS):
        
        # Your code: implement the training process and save the best model.
        
        train_loss, train_acc = train(train_dataloader, model, criterion, optimizer, lr_scheduler, device)
        valid_loss, valid_acc = evaluate(valid_dataloader, model, criterion, device)
        
        epoch_train_loss = np.mean(train_loss)
        epoch_train_acc = np.mean(train_acc)
        epoch_valid_loss = np.mean(valid_loss)
        epoch_valid_acc = np.mean(valid_acc)

        # Save the model that achieves the smallest validation loss.
        if epoch_valid_loss < best_valid_loss:
            # Your code: save the best model somewhere (no need to submit it to Sakai)
            best_valid_loss = epoch_valid_loss
            if not os.path.exists(CHECKPOINT_FOLDER):
               os.makedirs(CHECKPOINT_FOLDER)
            print("Saving ...")
            torch.save(model, os.path.join(CHECKPOINT_FOLDER, f'{model_type}.pth'))

        print(f'epoch: {epoch+1}')
        print(f'train_loss: {epoch_train_loss:.3f}, train_acc: {epoch_train_acc:.3f}')
        print(f'valid_loss: {epoch_valid_loss:.3f}, valid_acc: {epoch_valid_acc:.3f}')


    # Your Code: Load the best model's weights.
    model =torch.load(os.path.join(CHECKPOINT_FOLDER, f'{model_type}.pth'))
    model.eval()

    # Your Code: evaluate test loss on testing dataset (NOT Validation)
    test_loss, test_acc = evaluate(test_dataloader, model, criterion, device)

    epoch_test_loss = np.mean(test_loss)
    epoch_test_acc = np.mean(test_acc)
    print(f'test_loss: {epoch_test_loss:.3f}, test_acc: {epoch_test_acc:.3f}')
    
    # Free memory for later usage.
    del model
    torch.cuda.empty_cache()
    return {
        "test_loss": epoch_test_loss,
        "test_acc": epoch_test_acc,
    }