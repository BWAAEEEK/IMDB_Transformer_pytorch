import pickle
from tqdm import tqdm

with open("./tokenized_data", "rb") as f:
    data = pickle.load(f)

with open("./vocab", "rb") as f:
    vocab = pickle.load(f)

with open("./imdb.pickle", "rb") as f:
    imdb = pickle.load(f)

f.close()

print(data["text"][0])
print(data["label"][0])

print(imdb["text"][0])


def word_to_num(data, idx, max_len):
    seq = []
    seq.append(vocab["stoi"]["[sos]"])
    for word in data["text"][idx]:
        seq.append(vocab["stoi"][word])

    if len(seq) < max_len:
        for _ in range(max_len - len(seq)):
            seq.append(vocab["stoi"]["[pad]"])
    else:
        del seq[500:]

    return seq


# for i in data["label"]:
#     if i != 0:
#         print(i)

print(len(word_to_num(data, 0, 500)))

for i in word_to_num(data, 0, 500):
    print(vocab["itos"][i], end=" ")

label_list = set()

for i in data["label"]:
    print(i)
    label_list.add(i)


print(label_list)