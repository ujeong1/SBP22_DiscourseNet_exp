import pickle
import networkx as nx
dataset = "FBAD"
feature = "bert"
G = nx.read_edgelist("../"+dataset+"_"+feature+"_edgelist.gz")
cliques = list(nx.find_cliques(G))
hypergraph = dict()
clique_size = 3
for i, clique in enumerate(cliques):
    if len(clique) < clique_size:
        continue
    key = "clique_"+str(i)
    hypergraph[key] = [int(idx) for idx in clique]

with open("../"+dataset+"_"+feature+"_clique.pkl", "wb") as f:
    pickle.dump(hypergraph, f)
with open("../"+dataset+"_"+feature+"_clique.pkl", "rb") as f:
    data = pickle.load(f)
    print(data, len(data))
