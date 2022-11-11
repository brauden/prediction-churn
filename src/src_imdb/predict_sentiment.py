import torch

def tokenize(vocab: dict, example: str)-> list:
    """
    Tokenize the give example string into a list of token indices.
    :param vocab: dict, the vocabulary.
    :param example: a string of text.
    :return: a list of token indices.
    """
    # Your code here.
    lst_inx = [vocab[word] if word in vocab.keys() else 1 for word in example.lower().split() ]
    return lst_inx

def predict_sentiment(text, model, vocab, device):
    ids = tokenize(vocab, text)
    # ids = [vocab[t] if t in vocab else 1 for t in tokens]
    length = torch.LongTensor([len(ids)])
    tensor = torch.LongTensor(ids).unsqueeze(dim=0).to(device)
    prediction = model(tensor, length).squeeze(dim=0)
    probability = torch.softmax(prediction, dim=-1)
    predicted_class = prediction.argmax(dim=-1).item()
    predicted_probability = probability[predicted_class].item()
    return predicted_class, predicted_probability

if __name__ == '__main__':
    predict_sentiment()