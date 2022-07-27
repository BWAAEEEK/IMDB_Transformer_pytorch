from datasets import load_dataset
import pickle


# loading dataset
imdb = load_dataset("imdb")

# check dataset
print(imdb["train"]["text"][0])
print(imdb["train"]["label"][0])

# concat data
data = {"text": imdb["train"]["text"] + imdb["test"]["text"], "label": imdb["train"]["label"] + imdb["test"]["label"]}

# save dataset as pickle file
print("saving dataset ...")
with open("./imdb.pickle", "wb") as f:
    pickle.dump(data, f)

print("complete ...")


