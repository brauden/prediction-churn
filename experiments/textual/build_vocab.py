from collections import Counter


def build_vocab(x_train: list, min_freq: int = 5, hparams=None) -> dict:
    """
    build a vocabulary based on the training corpus.
    :param x_train:  List. The training corpus. Each sample in the list is a string of text.
    :param min_freq: Int. The frequency threshold for selecting words.
    :return: dictionary {word:index}
    """
    # Add your code here. Your code should assign corpus with a list of words.
    x_train_corpus = Counter(
        [
            word.lower()
            for sent in x_train
            for word in sent.split()
            if word.lower() not in hparams.STOP_WORDS or word.lower() != " "
        ]
    )

    corpus = []  # placeholder
    # sorting on the basis of most common words
    # corpus_ = sorted(corpus, key=corpus.get, reverse=True)[:1000]
    corpus_ = [
        word.lower() for word, freq in x_train_corpus.items() if freq >= min_freq
    ]
    # creating a dict
    vocab = {w: i + 2 for i, w in enumerate(corpus_)}
    vocab[hparams.PAD_TOKEN] = hparams.PAD_INDEX
    vocab[hparams.UNK_TOKEN] = hparams.UNK_INDEX
    return vocab


if __name__ == "__main__":
    build_vocab()
