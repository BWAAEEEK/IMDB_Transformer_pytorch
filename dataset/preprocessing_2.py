import pickle
from tqdm import tqdm
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

class Preprocessing:

    def __init__(self, data):

        self.text = list(data["text"])
        self.label = list(data["label"])

        self.dataset = {"text": [], "label": [i for i in self.label]}
        self.vocab = {}

    def tokenize(self):

        for text in tqdm(self.text):
            # filter
            tag_filter = re.compile("(<([^>]+)>)")
            punc_filter = re.compile("[\W\d]")
            short_filter = re.compile(r'\W*\b\w{1,2}\b')

            # Cleaning and Normalization
            text = text.lower()
            text = re.sub(short_filter, "", text)

            # Stemmer
            stem = PorterStemmer()

            # tokenize word in sentence
            text = re.sub(tag_filter, "", text)

            # stopword list
            stopword_list = stopwords.words("english")

            # split sentence for token
            token_list = text.split(" ")

            filtered_sen = list()
            for token in token_list:
                if token not in stopword_list:
                    tmp = re.sub(punc_filter, "", token)  # delete punctuation and number
                    tmp = stem.stem(tmp)  # stemming

                    if tmp != '':
                        filtered_sen.append(tmp)

            self.dataset["text"].append(filtered_sen)

    def construct_vocab(self):
        sentence_list = self.dataset["text"]

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

        vocab_list = ["[pad]", "[sos]", "[unk]"] + list(seq[0] for seq in word_freq)

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
