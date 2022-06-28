import pandas as pd
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import networkx as nx


def preprocess_document(document, sentence_spliter='.', word_spliter=' ', punct_mark=','):
    # lowercase all words and remove trailing whitespaces
    document = document.lower().strip()

    # remove unwanted punctuation marks
    for pm in punct_mark:
        document = document.replace(pm, '')

    # get list of sentences which are non-empty
    sentences = [sent for sent in document.split(sentence_spliter) if sent != '']

    # get list of sentences which are lists of words
    document = []
    for sent in sentences:
        words = sent.strip().split(word_spliter)
        document.append(words)

    return document


def get_entities(document):
    # in our case, entities are all unique words
    unique_words = []
    for sent in document:
        for word in sent:
            if word not in unique_words:
                unique_words.append(word)
    return unique_words


def get_relations(document):
    # in our case, relations are bigrams in sentences
    bigrams = []
    for sent in document:
        for i in range(len(sent) - 1):
            # for every word and the next in the sentence
            pair = [sent[i], sent[i + 1]]
            # only add unique bigrams
            if pair not in bigrams:
                bigrams.append(pair)
    return bigrams


def plot_graph(G, title=None):
    # set figure size
    plt.figure(figsize=(10, 10))

    # define position of nodes in figure
    pos = nx.nx_agraph.graphviz_layout(G)

    # draw nodes and edges
    nx.draw(G, pos=pos, with_labels=True)

    # get edge labels (if any)
    edge_labels = nx.get_edge_attributes(G, 'weight')

    # draw edge labels (if any)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    # plot the title (if any)
    plt.title(title)

    plt.show()
    return


def get_weighted_edges(document):
    # in our case, relations are bigrams in sentences
    # weights are number of equal bigrams
    # use a dict to store number of counts
    bigrams = {}
    for sent in document:
        for i in range(len(sent) - 1):

            # transform to hashable key in dict
            pair = str([sent[i], sent[i + 1]])

            if pair not in bigrams.keys():
                # weight = 1
                bigrams[pair] = 1
            else:
                # already exists, weight + 1
                bigrams[pair] += 1

    # convert to NetworkX standard form each edge connecting nodes u and v = [u, v, weight]
    weighted_edges_format = []
    for pair, weight in bigrams.items():
        # revert back from hashable format
        w1, w2 = eval(pair)
        weighted_edges_format.append([w1, w2, weight])

    return weighted_edges_format


def build_weighted_digraph(document):
    # preprocess document for standardization
    pdoc = preprocess_document(doc)

    # get graph nodes
    nodes = get_entities(pdoc)

    # get weighted edges
    weighted_edges = get_weighted_edges(pdoc)

    # create graph structure with NetworkX
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_weighted_edges_from(weighted_edges)

    return G


mypath = "processed_ad/"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
csv = {}
for filename in onlyfiles:
    keyword = filename.split(",")[1]
    df = pd.read_csv(mypath + filename)
    df = df.drop('Unnamed: 0', axis=1)
    csv[keyword] = df
    break;

target = csv[keyword]
print(keyword)
print(target)
for idx, row in target.iterrows():
    document = row["ad_creative_body"]
    words = preprocess_document(document)
    entities = get_entities(words)
    relations = get_relations(words)
    print(entities)
    print(relations)
    break
    # G = nx.Graph()