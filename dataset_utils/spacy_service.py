import re
import pandas as pd
import spacy
import numpy as np
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

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
    stemmer = WordNetLemmatizer()

stop_words = stopwords.words('english')
stop_words = set(stop_words)
stemmer = WordNetLemmatizer()
dataset = "covid"
nlp = spacy.load("en_core_web_lg")
df = pd.read_csv("../dataset/sbp_dataset.csv")
text =df.ad_creative_body.values.tolist()
vectors = []
for line in text:
    #line = line.strip().decode('latin1')
    '''
    temp = line
    line = word_tokenize(clean_str(line))
    line = ' '.join([stemmer.lemmatize(word) for word in line])
    if doc == None:
        doc = temp
    '''
    doc = line.strip()#clean_str_simple_version(line)
    #doc tokenizer should be here
    doc = nlp(doc)
    vector = doc.vector
    vectors.append(vector)

result = np.array(vectors)
print(result.shape)
#np.savez(filename, feature=result)
with open("../"+dataset+'_spacy.npy', 'wb') as f:
    np.save(f, result)
