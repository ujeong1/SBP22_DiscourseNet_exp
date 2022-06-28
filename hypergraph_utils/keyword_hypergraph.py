import pandas as pd
from rake_nltk import Rake
from tqdm import tqdm
import pickle
'''
dataset = "R8"
text = []
with open("../" +dataset+"_corpus.txt", "rb") as f:
    for line in f.readlines():
        line = line.decode('latin1')
        text.append(line)
'''
#text = pd.read_csv("../covid_dataset_shuffled.csv")["processed_text"].values
#text = pd.read_csv("../dataset/sbp_dataset.csv")["processed_text"].values
text = pd.read_csv("../dataset/sbp_dataset.csv")["ad_creative_body"].values
r = Rake()

K = []
for my_text in text:
    # my_text = "I ve experienced covid firsthand and lost people close to me but I still believe in and will defend freedom and choice I was proud to vote against vaccine mandates in committee yesterday Here is my speech"
    r.extract_keywords_from_text(my_text)
    keywordList = []
    rankedList = r.get_ranked_phrases_with_scores()
    print(rankedList)
    for keyword in rankedList:
        keyword_updated = keyword[1].split()
        window_size = 5
        
        for i in range(1, window_size+1):
            keyword_updated_string = " ".join(keyword_updated[:i])
            if len(keyword_updated_string.split()) < 2:
                continue
            keywordList.append(keyword_updated_string)
        '''
        keyword_updated_string = " ".join(keyword_updated[:2])
        keywordList.append(keyword_updated_string)
        '''
        #if (len(keywordList) > 5):
        #    break
    keywordList = list(set(keywordList))
    print(keywordList)
    print("*"*10)
    K.append(keywordList)
hypergraph = dict()
for keywords in K:
    for keyword in keywords:
        hypergraph[keyword] = []

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
print(result)
with open("../covid_keyword.pkl", "wb") as f:
    pickle.dump(result, f)
