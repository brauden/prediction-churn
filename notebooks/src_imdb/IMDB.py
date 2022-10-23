from torch.utils.data import Dataset

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


class IMDB(Dataset):
    def __init__(self, x, y, vocab, max_length=256) -> None:
        """
        :param x: list of reviews
        :param y: list of labels
        :param vocab: vocabulary dictionary {word:index}.
        :param max_length: the maximum sequence length.
        """
        self.x = x
        self.y = y
        self.vocab = vocab
        self.max_length = max_length

    def __getitem__(self, idx: int):
        """
        Return the tokenized review and label by the given index.
        :param idx: index of the sample.
        :return: a dictionary containing three keys: 'ids', 'length', 'label' which represent the list of token ids, the length of the sequence, the binary label.
        """
        # Add your code here.
        dic_token = {}
        lst_token = tokenize(self.vocab,self.x[idx])
        lst_token = lst_token[:self.max_length]
        dic_token['ids'] = lst_token
        dic_token['label'] = self.y[idx] 
        dic_token['length'] = len(lst_token)

        return dic_token

    def __len__(self) -> int:
        return len(self.x)