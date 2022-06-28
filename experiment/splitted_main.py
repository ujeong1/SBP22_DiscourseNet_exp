import copy
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
from sklearn import metrics
import argparse
import random
from utils import generate_G_from_H, get_hypergraph
from model import DualOptim, GraphConv, HyperGAT, MLP
# from RL import get_NMI, RLagent
from sklearn.utils import class_weight
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score
device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='GAT', choices=['GCN', 'GAT', 'SAGE'])
parser.add_argument('--feature', type=str, default='bert',
                    choices=['spacy', 'bert'])
parser.add_argument('--dataset', type=str, default='covid')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--train_ratio', type=float, default=1.00, help='dropout rate')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--l2', type=float, default=0.01, help='l2 penalty')
parser.add_argument('--lr_dc', type=float, default=0.5, help='learning rate decay')
parser.add_argument('--lr_dc_step', type=int, default=20, help='default')
parser.add_argument('--rand', type=int, default=777, help='random seed')
parser.add_argument('--epoch', type=int, default=30, help='epoch size')
parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
parser.add_argument('--hidden_channels', type=int, default=128, help='default')
parser.add_argument('--use_keyword', action='store_true', help='')
parser.add_argument('--use_entity', action='store_true', help='')
parser.add_argument('--use_clique', action='store_true', help='')

args = parser.parse_args()
print(args)

SEED = args.rand
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
if args.feature == "bert":
    data_nodes = np.load(args.dataset + "_dataset_bert.npy")
    args.in_channels = 512
# if args.feature == "covid":
labels = ["Prevention", "Treatment", "Diagnosis", "Mechanism", "Case Report", "Transmission", "Forecasting", "General"]

# elif args.feature == "spacy":
#     data_nodes = np.load(args.dataset + "_spacy.npy")
#     args.in_channels = 300


def label2class(dataset):
    labels = []
    # with open(dataset + "_labels.txt", "r") as f:
    #     for line in f.readlines():
    #         label = line.split("\t")[2].strip()
    #         if label not in labels:
    #             labels.append(label)
    labels = ["Prevention", "Treatment", "Diagnosis", "Mechanism", "Case Report", "Transmission", "Forecasting",
              "General"]
    cls = dict()
    for i, label in enumerate(set(labels)):
        cls[label] = i

    # from collections import Counter
    # cls = result.values()
    # print("Class distribution: ", Counter(cls))
    return cls


def shuffle_dict(d):
    keys = list(d.keys())
    random.shuffle(keys)

    result = dict()
    for key in keys:
        result.update({key: d[key]})
    return result
def crossval(dataset, cls):
    df = pd.read_csv("dataset/covid_dataset.csv")
    num_nodes = len(df)
    targets = []
    for i, row in df.iterrows():
        target = []
        for label in labels:
            if row[label] == 1:
                target.append(1)
            else:
                target.append(0)
        targets.append(target)

    # simply use the first 90% as training set
    ratio = 0.9
    origin_idx = np.arange(num_nodes)
    np.random.shuffle(origin_idx)

    rounds = []
    num_val = int(num_nodes * 0.2)
    for step in range(5):
        start = int((step*0.2) * num_nodes)
        end = min(int(start + num_val), num_nodes)
        val_idx = origin_idx[start:end].tolist()
        train_idx = origin_idx.copy()
        train_idx = np.delete(train_idx, np.arange(start, end), None).tolist()
        rounds.append((train_idx, val_idx))
    return rounds



def split_data(dataset, cls):
    df = pd.read_csv("dataset/covid_dataset.csv")
    num_nodes = len(df)
    targets = []
    for i, row in df.iterrows():
        target = []
        for label in labels:
            if row[label] == 1:
                target.append(1)
            else:
                target.append(0)
        targets.append(target)

    # simply use the first 90% as training set
    ratio = 0.9
    origin_idx = np.arange(num_nodes)
    np.random.shuffle(origin_idx)

    num_train = int(num_nodes * ratio)
    train_idx = origin_idx[np.arange(num_train)]
    test_idx = origin_idx[np.arange(num_train + 1, num_nodes)]
    train_idx = train_idx.tolist()
    test_idx = test_idx.tolist()
    origin_idx = origin_idx.tolist()
    train_data = dict()
    for idx in train_idx:
        train_data[idx] = targets[idx]
    test_data = dict()
    for idx in test_idx:
        test_data[idx] = targets[idx]
    origin_data = dict()
    for idx in origin_idx:
        origin_data[idx] = targets[idx]

    # extract valid set from train dataset
    num_valid = int(len(train_idx) * 0.1)
    valid_idx = random.sample(train_idx, k=num_valid)
    valid_data = dict()
    for idx in train_idx:
        if idx in valid_idx:
            valid_data[idx] = train_data[idx]
    for idx in valid_idx:
        del train_data[idx]

    # train_data = shuffle_dict(train_data)
    return train_data, valid_data, test_data, origin_data


def reduce_train(train_data, ratio):
    keys = train_data.keys()
    keys = random.sample(keys, int(len(keys) * ratio))
    result = dict()
    for key in keys:
        result[key] = train_data[key]
    # from collections import Counter
    # cls = result.values()
    # print("Class distribution: ", Counter(cls))
    return result


def class_ratio(train_data):
    train_y = list(train_data.values())
    class_weights = class_weight.compute_class_weight('balanced', np.unique(train_y), train_y)
    return class_weights


def numpy2tensor(nodes, edges, labels, HT):
    nodes = torch.Tensor(nodes).to(device)
    edges = torch.LongTensor(edges).to(device)
    labels = torch.LongTensor(labels).to(device)
    HT = torch.Tensor(HT).to(device)
    return nodes, edges, labels, HT


def class2idx(cls_dict, origin_data):
    classes = list(cls_dict.values())
    cls2idx_dict = dict()
    for cls in classes:
        cls2idx_dict[cls] = []
    for idx, cls in origin_data.items():
        cls2idx_dict[cls].append(idx)
    return cls2idx_dict


def label2idx(cls_dict, origin_data):
    labels = list(cls_dict.keys())
    label2idx_dict = dict()
    for label in labels:
        label2idx_dict[label] = []
    for idx, cls in origin_data.items():
        label = labels[cls]
        label2idx_dict[label].append(idx)
    return label2idx_dict


cls_dict = label2class(args.dataset)
rounds = crossval(args.dataset, cls_dict)
train_data, valid_data, test_data, origin_data = split_data(args.dataset, cls_dict)
train_data = reduce_train(train_data, args.train_ratio)
# class_weights = class_ratio(train_data)
train_idx = list(train_data.keys())
valid_idx = list(valid_data.keys())
test_idx = list(test_data.keys())

print("Data Split")
print(len(train_idx), len(valid_idx), len(test_idx))
# G = nx.read_edgelist(args.dataset + "_" + args.feature + "_edgelist.gz")
# data_edge_index = [[int(edge[0]), int(edge[1])] for edge in G.edges]
# data_edge_index = np.array(data_edge_index).T
data_edge_index = np.arange(10)
data_labels = np.array(list(origin_data.values()))
# args.num_classes = len(data_labels.unique())
args.num_classes = 8
num_nodes = len(data_labels)
print("Number of news: ", num_nodes)
HT = get_hypergraph(args.dataset, args.feature, num_nodes, args)
data_nodes, data_edge_index, data_labels, HT = numpy2tensor(data_nodes, data_edge_index, data_labels, HT)
print("* Hypergraph Size", HT.shape)

# H = np.transpose(np.squeeze(np.array(HT), axis=0)).astype(np.float32)
# HG = generate_G_from_H(H)

graph_model = GraphConv(args).to(device)
hypergraph_model = HyperGAT(args).to(device)
# model = DualOptim(args, graph_model, hypergraph_model).to(device)
model = MLP(args, graph_model, hypergraph_model).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)


# loss_function = nn.CrossEntropyLoss()  # weight = torch.Tensor(class_weights).float().to(device))
def loss_fn(outputs, targets):
    return nn.BCEWithLogitsLoss()(outputs, targets)

def train(train_idx):
    model.train()
    total_loss = 0
    num_train = len(train_idx)

    # losses = []
    # checkpoint_losses = []
    # n_total_steps = num_train

    for idx in range(0, num_train, args.batchSize):
        slices = train_idx[idx:min(idx + args.batchSize, num_train)]
        optimizer.zero_grad()
        outputs, edge = model(data_nodes, data_edge_index, HT)
        # out = model.compute_score(out)
        # scores = out[slices]
        # labels = data_labels[slices]
        # loss = loss_function(scores, labels)
        outputs = outputs[slices]
        labels = data_labels[slices].to(device, dtype=torch.float)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        # total_loss += float(loss) * len(slices)
        total_loss = total_loss + (1 / (idx + 1)*(loss.item() - total_loss))
        # losses.append(loss.item())
    # return total_loss / num_train
    return total_loss


@torch.no_grad()
def test(test_idx, verbose=False, save=False):
    model.eval()
    num_samples = len(test_idx)
    args.batchSize = num_samples
    # if save:
    #     out, node2edge = model(data_nodes, data_edge_index, HT)
    #     node2edge = node2edge.squeeze(0)[test_idx]
    #     node2edge = torch.transpose(node2edge, 0, 1)
    #     torch.save(node2edge, args.dataset + "_" + args.label + '_edge.pt')
    #     return 0, 0
    args.batchSize = num_samples
    acc, details = 0, 0
    for idx in range(0, num_samples, args.batchSize):
        slices = test_idx[idx:min(idx + args.batchSize, num_samples)]
        outputs, edge = model(data_nodes, data_edge_index, HT)
        outputs = outputs[slices]
        labels = data_labels[slices]
        outputs = torch.sigmoid(outputs).cpu()
        preds = np.round(outputs)
        if save:
            for pred, label in zip(preds, labels):
                print(pred, label)
        # total += labels.size(0)
        # correct += (preds == labels).sum().item()
        acc = accuracy_score(preds, labels)
    # acc = 100 * correct / total

    return acc, details


args.train = True
if args.train:
    best_val_acc = 0
    for epoch in range(1, args.epoch):

        # print("NMI before")
        # train_nodes = data_nodes[train_idx].detach().numpy()
        # nmi = get_NMI(train_nodes, train_data, args.num_classes)
        # print(nmi)

        loss = train(train_idx)

        # print("NMI after")
        # train_nodes = model(data_nodes, data_edge_index, HT)[train_idx].detach().numpy()
        # nmi = get_NMI(train_nodes, train_data, args.num_classes)
        # print(nmi)
        # threshold = RLagent()

        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')
        val_acc, _ = test(valid_idx, verbose=False)
        print(f'Val Accuracy: {val_acc:.4f}')
        if best_val_acc <= val_acc:
            best_val_acc = val_acc
            state_dict = copy.deepcopy(model.state_dict())
    print("** The Best model on test dataset: ")
    torch.save(state_dict, args.dataset + "_model.pt")

print(test_idx)
state_dict = torch.load(args.dataset + "_model.pt")
model.load_state_dict(state_dict)
acc, details = test(test_idx, verbose=False, save=True)
print("accuracy:", acc)
exit(0)

# label2idx_dict = label2idx(cls_dict, origin_data)
label2idx_dict = {"Prevention":0, "Treatment":1, "Diagnosis":2, "Mechanism":3, "Case Report":4, "Transmission":5, "Forecasting":6, "General":7}

args.attention = False

if args.attention:
    with open(args.dataset + "_label2idx.pkl", "wb") as f:
        pickle.dump(label2idx_dict, f)
    for label in label2idx_dict.keys():
        args.label = label
        attention_idx = label2idx_dict[label]
        acc, details = test(attention_idx, verbose=False, save=True)

