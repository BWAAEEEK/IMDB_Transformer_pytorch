import pickle
from tqdm import tqdm
import re

class Preprocessing:

    def __init__(self, data):

        self.text = list(data["unsupervised"]["text"])
        self.label = list(data["unsupervised"]["label"])

        self.dataset = {"text": [], "label": int}
        self.vocab = {}

    def tokenize(self):

        for text in tqdm(self.text):
            # tokenize word in sentence
            token_list = text.split(" ")

            # filtering
            tag_filter = re.compile("(<([^>]+)>)[\W\d]")

            filtered_sen = list()
            for token in token_list:
                filtered_sen.append(re.sub(tag_filter, "", token))

            self.dataset["text"].append(filtered_sen)

    def construct_vocab(self):
        sentence_list = list(self.dataset["text"])

        word_list = set()

        for sentence in tqdm(sentence_list):
            for word in sentence:
                word_list.add(word)

        # calculate word frequency
        word_count = {word: 0 for word in word_list}
        for sentence in tqdm(sentence_list):
            for word in sentence:
                word_count[word] += 1

        word_freq = sorted(word_count.items(), key=lambda x: x[1], reverse=True)

        vocab_list = ["[pad]", "[unk]"] + list(seq[0] for seq in word_freq)

        self.vocab = {
            "size": len(vocab_list),
            "itos": {i: word for i, word in enumerate(vocab_list)},
            "stoi": {word: i for i, word in enumerate(vocab_list)}
        }

    def save(self):
        with open("./tokenized_data", "wb") as f:
            pickle.dump(self.dataset, f)

        with open("./vocab", "wb") as f:
            pickle.dump(self.vocab, f)


print("loading imdb dataset ...")

with open("./imdb.pickle", "rb") as f:
    imdb = pickle.load(f)
print("--- loading complete ---")

print("preprocessing ...")

setting = Preprocessing(imdb)

print("tokenize")
setting.tokenize()

print("construct vocab ")
setting.construct_vocab()

print("save data ...")
setting.save()

print("--- Program Complete ---")

