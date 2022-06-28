import torch
import pickle
import scipy.sparse as sp
import numpy as np


def _generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    H = np.array(H)
    n_edge = H.shape[1]
    # the weight of the hyperedge
    W = np.ones(n_edge)
    # the degree of the node
    DV = np.sum(H * W, axis=1)
    # the degree of the hyperedge
    DE = np.sum(H, axis=0)

    invDE = np.mat(np.diag(np.power(DE, -1)))
    DV2 = np.mat(np.diag(np.power(DV, -0.5)))
    W = np.mat(np.diag(W))
    H = np.mat(H)
    HT = H.T

    if variable_weight:
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        return DV2_H, W, invDE_HT_DV2
    else:
        G = DV2 * H * W * invDE * HT * DV2
        return G


def generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    if type(H) != list:
        return _generate_G_from_H(H, variable_weight)
    else:
        G = []
        for sub_H in H:
            G.append(generate_G_from_H(sub_H, variable_weight))
        return G


def get_adj_matrix(hyperedges, nodes_seq):
    items, n_node, HT, alias_inputs, node_masks, node_dic = [], [], [], [], [], []

    # In case we haven't extracted node sequence numbers
    # node_list = list(chain.from_iterable(hyperedges))

    node_list = list(nodes_seq)
    node_set = list(set(node_list))
    node_dic = {node_set[i]: i for i in range(len(node_set))}

    # This follows the the definition of hyperedge should at least contain 2 nodes
    # hyperedges = [hyperedge for hyperedge in hyperedges if len(hyperedge) > 1]
    rows = []
    cols = []
    vals = []
    max_n_node = len(node_set)
    max_n_edge = len(hyperedges)
    total_num_node = len(node_set)

    # num_hypergraphs can be used for batching different size of hypergraphs for training
    num_hypergraphs = 1
    for idx in range(num_hypergraphs):
        # e.g., hypergraph = [[12, 31, 111, 232],[12, 31, 111, 232],[12, 31, 111, 232] ...]
        for hyperedge_seq, hyperedge in enumerate(hyperedges):
            # e.g., hyperedge = [12, 31, 111, 232]
            for node_id in hyperedge:
                rows.append(node_dic[node_id])
                cols.append(hyperedge_seq)
                vals.append(1)
        u_H = sp.coo_matrix((vals, (rows, cols)), shape=(max_n_node, max_n_edge))
        HT.append(np.asarray(u_H.T.todense()))
        alias_inputs.append([j for j in range(max_n_node)])
        node_masks.append([1 for j in range(total_num_node)] + (max_n_node - total_num_node) * [0])

    return alias_inputs, HT, node_masks


def get_hypergraph(dataset, feature, num_nodes, args):
    hypergraph = dict()
    if args.use_entity:
        with open(dataset + "_entity.pkl", "rb") as f:
           e_hypergraph = pickle.load(f)
        hypergraph.update(e_hypergraph)
    if args.use_advertiser:
        with open(dataset + "_advertiser.pkl", "rb") as f:
           a_hypergraph = pickle.load(f)
        hypergraph.update(a_hypergraph)
    if args.use_keyword:
        with open(dataset + "_keyword.pkl", "rb") as f:
            k_hypergraph = pickle.load(f)
        hypergraph.update(k_hypergraph)
    if args.use_clique:
        with open(dataset + "_" + feature + "_clique.pkl", "rb") as f:
            c_hypergraph = pickle.load(f)
        hypergraph.update(c_hypergraph)
    values = []
    for hyperedge in hypergraph.values():
        values+=hyperedge
    values = set(values)
    print("node coverage: ", len(values))

    nodes_seq = np.arange(num_nodes)
    threshold = 2
    hyperedges = [hyperedge for hyperedge in hypergraph.values() if len(hyperedge) >= threshold]
    alias_inputs, HT, node_masks = get_adj_matrix(hyperedges, nodes_seq)
    return HT
