import spacy
from tqdm import tqdm
from pathlib import Path
from spacy.tokens import DocBin, Doc
# from spacy.training.example import Example
from scripts.rel_pipe import make_relation_extractor, score_relations
from scripts.rel_model import create_relation_model, create_classification_layer, create_instances, create_tensors
import pandas as pd
import re
import pickle


def get_labels():
    labels = ['ORG', 'CARDINAL', 'DATE', 'GPE', 'PERSON', 'MONEY', 'PRODUCT', 'TIME', 'PERCENT', 'WORK_OF_ART',
              'QUANTITY', 'NORP', 'LOC', 'EVENT', 'ORDINAL', 'FAC', 'LAW', 'LANGUAGE']
    labels.remove("MONEY")
    labels.remove("PERCENT")
    labels.remove("QUANTITY")
    labels.remove("ORDINAL")
    labels.remove("CARDINAL")
    labels.remove("TIME")
    labels.remove("FAC")
    labels.remove("LANGUAGE")
    labels.remove("LAW")
    labels.remove("DATE")

    '''
    PERSON:      People, including fictional.
    NORP:        Nationalities or religious or political groups.
    FAC:         Buildings, airports, highways, bridges, etc.
    ORG:         Companies, agencies, institutions, etc.
    GPE:         Countries, cities, states.
    LOC:         Non-GPE locations, mountain ranges, bodies of water.
    PRODUCT:     Objects, vehicles, foods, etc. (Not services.)
    EVENT:       Named hurricanes, battles, wars, sports events, etc.
    WORK_OF_ART: Titles of books, songs, etc.
    LAW:         Named documents made into laws.
    LANGUAGE:    Any named language.
    DATE:        Absolute or relative dates or periods.
    TIME:        Times smaller than a day.
    PERCENT:     Percentage, including ”%“.
    MONEY:       Monetary values, including unit.
    QUANTITY:    Measurements, as of weight or distance.
    ORDINAL:     “first”, “second”, etc.
    CARDINAL:    Numerals that do not fall under another type.
    '''
    return labels


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def clean_str_simple_version(string):
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()


nlp = spacy.load('en_core_web_trf')
docs = []
dataset = "R8"
filename = "../" + dataset + "_corpus.txt"
csvs = dict()
docs = []
with open(filename, 'rb') as f:
    for line in f.readlines():
        doc = line.strip().decode('latin1')
        doc = clean_str_simple_version(doc)
        docs.append(doc)
E = []
i = 0

print("************NER")
labels = get_labels()
print(labels)
for doc in nlp.pipe(docs, disable=["tagger", "parser", "tok2vec", "attribute_ruler"]):
    entities = dict()
    for e in doc.ents:
        if e.label_ in labels:
            entities[e.text] = e.label_
    print(entities)
    E.append(entities)
hypergraph = dict()
for entities in E:
    for text in entities.keys():
        hypergraph[text] = []

print("************Hypergraph")
for entities1 in tqdm(E):
    for text in entities1.keys():
        hyperedge = []
        for idx, entities2 in enumerate(E):
            if text in list(entities2.keys()):
                hyperedge.append(idx)
        hypergraph[text] += hyperedge

result = dict()
for key, hyperedge in hypergraph.items():
    hyperedge = list(set(hyperedge))
    if len(hyperedge) >= 2:
        result[key] = hyperedge

print("************Saving Pickle")
with open("../"+dataset + "_entity.pkl", "wb") as f:
    pickle.dump(result, f)

#with open(dataset + "_hypergraph.pkl", "rb") as f:
#    data = pickle.load(f)
#print("file name", dataset + "_hypergraph.pkl")
#print(data)
