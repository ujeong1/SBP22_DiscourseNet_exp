import pandas as pd
from tqdm import tqdm
import pickle
from keybert import KeyBERT
from keyphrase_vectorizers import KeyphraseCountVectorizer

'''
dataset = "R8"
text = []
with open("../" +dataset+"_corpus.txt", "rb") as f:
    for line in f.readlines():
        line = line.decode('latin1')
        text.append(line)
'''
#text = pd.read_csv("../dataset/covid_dataset.csv")["text"].values
#text = list(pd.read_csv("../dataset/covid_final - covid_final.csv")["text"].values)
text = list(pd.read_csv("../dataset/covid_final.csv")["text"].values)

kw_model = KeyBERT()
vectorizer = KeyphraseCountVectorizer()
K = []

document_keyphrase_matrix = vectorizer.fit_transform(text).toarray()
keyphrases = vectorizer.get_feature_names_out()

#keywords = kw_model.extract_keywords(text)
#print(keywords)
#kw_model.extract_keywords(docs=docs, keyphrase_ngram_range=(1,2))

keywordList = kw_model.extract_keywords(docs=text, vectorizer=KeyphraseCountVectorizer())
for keywords in keywordList:
    keywords = [keyword[0] for keyword in keywords]
    K.append(keywords)

hypergraph = dict()
for keywords in K:
    for keyword in keywords:
        hypergraph[keyword] = []
'''
for keyphrase in keyphrases:
    hypergraph[keyphrase] = []

for i, doc in enumerate(text):
    for keyphrase in keyphrases:
        if keyphrase in doc:
            hypergraph[keyphrase].append(i)

result = dict()
for key, hyperedge in hypergraph.items():
    hyperedge = list(set(hyperedge))
    if len(hyperedge) >= 2:
        result[key] = hyperedge
'''

print("************Hypergraph")
for keywords1 in tqdm(K):
    for text in keywords1:
        hyperedge = []
        for idx, keywords2 in enumerate(K):
            if text in list(keywords2):
                hyperedge.append(idx)
        hypergraph[text] += hyperedge

result = dict()
for key, hyperedge in hypergraph.items():
    hyperedge = list(set(hyperedge))
    if len(hyperedge) >= 2:
        result[key] = hyperedge

print("************Saving Pickle")
'''
with open("../" + dataset + "_keyword.pkl", "wb") as f:
    pickle.dump(result, f)
'''
#with open("../dataset/covid_keyword.pkl", "wb") as f:
#    pickle.dump(result, f)
print(result)
