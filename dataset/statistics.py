import pickle
import matplotlib.pyplot as plt
import re

with open("./imdb.pickle", "rb") as f:
    imdb = pickle.load(f)

p = re.compile("[\W\d]")
text = re.sub(p, '', imdb["train"]["text"][0])

print(imdb["train"]["text"][0])
print(text)