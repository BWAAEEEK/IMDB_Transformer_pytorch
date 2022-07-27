import torch
import random
from torch.utils.data import Dataset
import pickle

class CustomDataset(Dataset):
    def __init__(self, data, vocab, args):
        self.vocab = vocab
        self.data = data

        self.max_len = args.max_len

    def __len__(self):
        return len(self.data["label"])

    def __getitem__(self, item):
        data_x = self.word_to_num(item)
        data_y = self.data["label"][item]

        return {"input": torch.tensor(data_x),
                "label": torch.tensor(data_y),
                "input_position": torch.tensor(list(range(len(data_x))))}

    def word_to_num(self, idx):
        seq = []
        seq.append(self.vocab["stoi"]["[sos]"])
        for word in self.data["text"][idx]:
            seq.append(self.vocab["stoi"][word])

        if len(seq) < self.max_len:
            for _ in range(self.max_len - len(seq)):
                seq.append(self.vocab["stoi"]["[pad]"])
        else:
            del seq[self.max_len:]

        return seq
