import pickle
import networkx as nx
dataset = "covid"
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
result = dict()
for key, hyperedge in hypergraph.items():
    hyperedge = list(set(hyperedge))
    if len(hyperedge) >= 2:
        result[key] = hyperedge
print(result)
with open("../"+dataset+"_"+feature+"_clique.pkl", "wb") as f:
    pickle.dump(result, f)
