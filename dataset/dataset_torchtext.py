import string

from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe

def tokenize(input):
    input = input.lower()

    for p in string.punctuation:
        input = input.replace(p, " ")

    return input.strip().split()

def num2words(vocab, vec):
    return [vocab.itos[i] for i in vec]

def get_imdb(batch_size, max_len):
    TEXT = data.Field

