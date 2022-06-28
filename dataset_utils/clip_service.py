import numpy as np
from tqdm import tqdm
import re
from clip_client import Client

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

docs = []
# dataset = "20NG"
dirname = "processed_text/"
dataset="covid_dataset"
with open(dirname+dataset+".txt", 'r') as f:
    i = 0
    for line in tqdm(f.readlines()):
        # line = line.decode('latin1')
        doc = clean_str_simple_version(line)
        docs.append(doc)
print("total documents:", len(docs))
c = Client('grpc://0.0.0.0:51000')
r = c.encode(docs, show_progress=True)
with open(dataset+'_bert.npy', 'wb') as f:
    np.save(f, r)
