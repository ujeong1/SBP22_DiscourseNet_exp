from nltk.corpus import stopwords
import nltk
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
from baseline_utils.utils import clean_str, show_statisctic, clean_document, clean_str_simple_version
import collections
from collections import Counter
import random
import numpy as np
import pickle
import json
from nltk import tokenize
from sklearn.utils import class_weight


def read_file(dataset, train_idx, val_idx, origin_idx, df):
    doc_content_list = []
    doc_sentence_list = []
    lines = df.ad_creative_body.values.tolist()
    #print(df.ad_creative_body.isnull().values.any())
    for i, line in enumerate(lines):
        doc_content_list.append(line.strip())
        doc_sentence_list.append(tokenize.sent_tokenize(clean_str_simple_version(doc_content_list[-1], dataset)))
        num_token = len(tokenize.sent_tokenize(clean_str_simple_version(doc_content_list[-1], dataset)))
    doc_content_list = [sentence[0].split() for sentence in doc_sentence_list]
    # doc_content_list = clean_document(doc_sentence_list, dataset)

    max_num_sentence = show_statisctic(doc_content_list)

    labels_dic = {}

    word_freq = Counter()
    word_set = set()
    for doc_words in doc_content_list:
        for words in doc_words:
            for word in words:
                word_set.add(word)
                word_freq[word] += 1

    vocab = list(word_set)

    vocab_dic = {}
    for i in word_set:
        vocab_dic[i] = len(vocab_dic) + 1

    print('Total_number_of_words: ' + str(len(vocab)))
    print('Total_number_of_categories: ' + str(len(labels_dic)))

    labels = ["Prevention", "Treatment", "Diagnosis", "Mechanism", "Case Report", "Transmission", "Forecasting",
              "General"]
    targets = []
    for i, row in df.iterrows():
        target = []
        for label in labels:
            if row[label] == 1:
                target.append(1)
            else:
                target.append(0)
        targets.append(target)

    doc_train_list = []
    doc_val_list = []
    for idx in train_idx:
        label = targets[idx]
        doc = doc_content_list[idx]
        temp_doc = []
        for sentence in doc:
            temp = []
            for word in sentence:
                temp.append(vocab_dic[word])
            temp_doc.append(temp)
        row = (temp_doc, label)
        doc_train_list.append(row)
    for idx in val_idx:
        label = targets[idx]
        doc = doc_content_list[idx]
        temp_doc = []
        for sentence in doc:
            temp = []
            for word in sentence:
                temp.append(vocab_dic[word])
            temp_doc.append(temp)
        row = (temp_doc, label)
        doc_val_list.append(row)

    return doc_content_list, doc_train_list, doc_val_list, vocab_dic, max_num_sentence


def loadGloveModel(gloveFile, vocab_dic, matrix_len):
    print("Loading Glove Model")
    f = open(gloveFile, 'r')
    gloveModel = {}
    glove_embedding_dimension = 0
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        glove_embedding_dimension = len(splitLine[1:])
        embedding = np.array([float(val) for val in splitLine[1:]])
        gloveModel[word] = embedding

    words_found = 0
    weights_matrix = np.zeros((matrix_len, glove_embedding_dimension))
    weights_matrix[0] = np.zeros((glove_embedding_dimension,))

    for word in vocab_dic:
        if word in gloveModel:
            weights_matrix[vocab_dic[word]] = gloveModel[word]
            words_found += 1
        else:
            weights_matrix[vocab_dic[word]] = gloveModel['the']

    print("Total ", len(vocab_dic), " words")
    print("Done.", words_found, " words loaded from", gloveFile)

    return weights_matrix
