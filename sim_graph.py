import numpy as np
from sklearn.preprocessing import normalize
from sentence_transformers import util
import matplotlib.pyplot as plt
import networkx as nx
def cosine(x, y):
    ranks = util.cos_sim(x, y)
    return ranks.reshape(-1)
def dot_product(embedding, embeddings):
    embeddings = normalize(embeddings)
    ranks = np.inner(embeddings, embedding)
    return ranks
feature = 'bert'
dataset="covid"
def build_graph():
    if feature == "bert":
        with open(dataset+"_bert.npy", "rb") as f:
            embeddings = np.load(f)
        x = embeddings[0]
    elif feature == "spacy":
        #embeddings = np.load(dataset+"_spacy.npz")['feature']
        with open(dataset+"_spacy.npy", "rb") as f:
            embeddings = np.load(f)
        x = embeddings[0]
    G = nx.Graph()
    nodes = np.arange(len(embeddings))
    G.add_nodes_from(nodes)
    for i, x in enumerate(embeddings):
        ranks = cosine(x, embeddings).numpy()
        #candidates = np.where(ranks>=0.80)[0]
        #neighbors = candidates.tolist()
        neighbors = list(np.argsort(ranks)[::-1][:5])
        if i in neighbors:
           neighbors.remove(i)
           neighbors.append(np.argsort(ranks)[::-1][6])
        print(i, neighbors)
        for j in neighbors:
            edge = (i, j)
            G.add_edge(*edge)
    nx.write_edgelist(G, dataset+"_"+feature+"_edgelist.gz")
    return G
G = build_graph()

G = nx.read_edgelist(dataset+"_"+feature+"_edgelist.gz")
nx.draw(G, node_size=5, with_labels=True)
plt.savefig(dataset+"_edge.png")
plt.show()
