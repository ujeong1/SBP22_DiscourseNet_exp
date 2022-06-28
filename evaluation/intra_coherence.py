import numpy as np
import itertools
import spacy
import pandas as pd
from statistics import mean
nlp = spacy.load("en_core_web_lg")
def score(filename):
    df = pd.read_csv(filename+"_attns.csv")
    words = df["name"]
    words = list(words[:10].values)
    vectors = []
    for word in words:
        vector = nlp(word)
        vectors.append(vector)
    result = []
    for comb in itertools.combinations(vectors, 2):
        left = comb[0]
        right = comb[1]
        similarity = left.similarity(right)
        result.append(similarity)
    return mean(result)
