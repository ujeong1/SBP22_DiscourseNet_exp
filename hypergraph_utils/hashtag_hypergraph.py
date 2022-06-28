import pandas as pd
from tqdm import tqdm
import re
import pickle

'''
dataset = "R8"
text = []
with open("../" +dataset+"_corpus.txt", "rb") as f:
    for line in f.readlines():
        line = line.decode('latin1')
        text.append(line)
'''


def extract_hashtags(text):
    # the regular expression
    regex = "#(\w+)"

    # extracting the hashtags
    hashtag_list = re.findall(regex, text)
    return hashtag_list


# text = pd.read_csv("../dataset/covid_dataset.csv")["text"].values
text = pd.read_csv("../dataset/covid_final - covid_final.csv")["text"].values

H = []
for my_text in text:
    # my_text = "I ve experienced covid firsthand and lost people close to me but I still believe in and will defend freedom and choice I was proud to vote against vaccine mandates in committee yesterday Here is my speech"

    hashtag_lsit = extract_hashtags(my_text)
    print(hashtag_lsit)
    H.append(hashtag_lsit)

hypergraph = dict()
for keywords in H:
    for keyword in keywords:
        hypergraph[keyword] = []

print("************Hypergraph")
for keywords1 in tqdm(H):
    for text in keywords1:
        hyperedge = []
        for idx, keywords2 in enumerate(H):
            if text in list(keywords2):
                hyperedge.append(idx)
        hypergraph[text] += hyperedge

result = dict()
for key, hyperedge in hypergraph.items():
    hyperedge = list(set(hyperedge))
    if len(hyperedge) >= 2:
        result[key] = hyperedge

print("************Saving Pickle")
# with open("../" + dataset + "_keyword.pkl", "wb") as f:
#     pickle.dump(result, f)
# with open("../dataset/covid_keyword.pkl", "wb") as f:
#    pickle.dump(result, f)
print(result)
