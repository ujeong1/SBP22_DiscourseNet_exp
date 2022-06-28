import numpy as np
import spacy
import itertools
from itertools import permutations
import pandas as pd
from statistics import mean
nlp = spacy.load("en_core_web_lg")
def get_vectors(filename):
    df = pd.read_csv(filename+"_attns.csv")
    words = df["name"]
    words = list(words[:10].values)
    vectors = []
    for word in words:
        vector = nlp(word)
        vectors.append(vector)
    return vectors
def score(filename1, filename2):
    vectors1 = get_vectors(filename1)
    vectors2 = get_vectors(filename2)
    unique_combinations = []
    permut = itertools.permutations(vectors1, len(vectors2))
    for comb in permut:
        zipped = zip(comb, vectors2)
        unique_combinations.append(list(zipped))
    result = []
    for combs in unique_combinations:
        for comb in combs:
            left = comb[0]
            right = comb[1]
            similarity = left.similarity(right)
            result.append(similarity)
    return mean(result)

if __name__ == "__main__":
    print(score("ship", "earn"))
