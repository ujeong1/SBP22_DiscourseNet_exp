import networkx as nx
import torch
import torch.nn as nn
import copy
from utils import generate_G_from_H, get_hypergraph
from model import DualOptim, GraphConv, HyperGAT, MLP
import statistics
import pandas as pd
from sklearn.metrics import accuracy_score
import cls_idx
import argparse
from baseline_utils.utils import split_validation, Data
from baseline_utils.preprocess import *
import random
import os
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix, classification_report

device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='GAT', choices=['GCN', 'GAT', 'SAGE'])
parser.add_argument('--feature', type=str, default='bert',
                    choices=['spacy', 'bert'])
parser.add_argument('--dataset', type=str, default='covid')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--val_ratio', type=float, default=0.2, help='dropout rate')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--l2', type=float, default=0.01, help='l2 penalty')
parser.add_argument('--lr_dc', type=float, default=0.5, help='learning rate decay')
parser.add_argument('--lr_dc_step', type=int, default=20, help='default')
parser.add_argument('--rand', type=int, default=777, help='random seed')
parser.add_argument('--epoch', type=int, default=40, help='epoch size')
parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
parser.add_argument('--hidden_channels', type=int, default=128, help='default')
parser.add_argument('--use_keyword', action='store_true', help='')
parser.add_argument('--use_entity', action='store_true', help='')
parser.add_argument('--use_advertiser', action='store_true', help='')
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
    data_nodes = np.load(args.dataset + "_bert.npy")
    args.in_channels = 768
else:
    data_nodes = np.load(args.dataset + "_spacy.npy")
    args.in_channels = 300
# if args.feature == "covid":
labels = ["Prevention", "Treatment", "Diagnosis", "Mechanism", "Case Report", "Transmission", "Forecasting", "General"]
args.num_classes = 8


def crossval(df, ratio):
    num_nodes = len(df)
    # targets = []
    # for i, row in df.iterrows():
    #     target = []
    #     for label in labels:
    #         if row[label] == 1:
    #             target.append(1)
    #         elif  row[label] == 0:
    #             target.append(0)
    #         else:
    #             print("wrong values in label")
    #             exit(0)
    #     targets.append(target)

    origin_idx = np.arange(num_nodes)

    rounds = []
    num_val = int(num_nodes * ratio)
    steps = 5
    for step in range(steps):
        start = int((step * ratio) * num_nodes)
        end = min(int(start + num_val), num_nodes) + 1
        val_idx = origin_idx[start:end].tolist()
        train_idx = origin_idx.copy()
        train_idx = np.delete(train_idx, np.arange(start, end), None)
        np.random.shuffle(train_idx)
        train_idx = train_idx.tolist()
        rounds.append((train_idx, val_idx))
    return rounds, origin_idx


def split_data(train_idx, test_idx, origin_idx, targets):
    origin_data = dict()
    for idx in origin_idx:
        origin_data[idx] = targets[idx]

    train_data = dict()
    for idx in train_idx:
        train_data[idx] = targets[idx]

    test_data = dict()
    for idx in test_idx:
        test_data[idx] = targets[idx]
    return train_data, test_data, origin_data


def import_targets(df):
    targets = []
    for i, row in df.iterrows():
        target = []
        for label in labels:
            if row[label] == 1:
                target.append(1)
            else:
                target.append(0)
        targets.append(target)

    return targets


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


def numpy2tensor(nodes, labels, HT, data_edge_index):
    nodes = torch.Tensor(nodes).to(device)
    # nodes = torch.LongTensor(nodes).to(device)
    labels = torch.LongTensor(labels).to(device)
    data_edge_index = torch.LongTensor(data_edge_index).to(device)
    HT = torch.Tensor(HT).to(device)
    return nodes, labels, HT, data_edge_index


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


def loss_fn(outputs, targets):
    return nn.BCEWithLogitsLoss()(outputs, targets)


def train(train_idx, data_edge_index):
    model.train()
    total_loss = 0
    num_train = len(train_idx)

    for idx in range(0, num_train, args.batchSize):
        slices = train_idx[idx:min(idx + args.batchSize, num_train)]
        optimizer.zero_grad()
        # data_edge_index = np.arange(10)
        outputs, edge = model(data_nodes, data_edge_index, HT)
        outputs = outputs[slices]
        targets = data_labels[slices].to(device, dtype=torch.float)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss = total_loss + (1 / (idx + 1) * (loss.item() - total_loss))

    return total_loss


@torch.no_grad()
def test(test_idx, data_edge_index, verbose=False, save=False):
    model.eval()
    num_samples = len(test_idx)
    args.batchSize = num_samples
    acc_category = dict()
    for label in labels:
        acc_category[label] = 0

    outputs, edge = model(data_nodes, data_edge_index, HT)
    outputs = outputs[test_idx]
    targets = data_labels[test_idx]
    outputs = torch.sigmoid(outputs).cpu()
    preds = np.round(outputs)
    acc = accuracy_score(preds, targets)
    for i in range(len(targets)):
        for j, label in enumerate(labels):
            if preds[i][j] == targets[i][j]:
                acc_category[label] += 1
    print("ACCURACY: ",acc)
    for label in labels:
        acc_category[label] /= num_samples
    preds = preds
    targets = targets
    report = classification_report(preds, targets, target_names=labels, output_dict=True)
    print(classification_report(preds, targets, target_names=labels))
    f1 = report['macro avg']['f1-score']
    # acc = report['micro avg']['f1-score']

    return acc, f1


@torch.no_grad()
def test_cls(cls_idx, data_edge_index, cls_name):
    model.eval()
    outputs, edge = model(data_nodes, data_edge_index, HT)
    edge = edge[cls_idx]
    edge = edge.transpose(0, 1)
    torch.save(edge, "covid_" + cls_name + "_edge.pt")


def import_data(train_idx, val_idx, origin_idx, df):
    doc_content_list, doc_train_list, doc_val_list, vocab_dic, max_num_sentence = read_file(args.dataset, train_idx,
                                                                                            val_idx, origin_idx, df)
    pre_trained_weight = []
    if args.dataset != 'covid':
        gloveFile = 'data/glove.6B.300d.txt'
        if not os.path.exists(gloveFile):
            print('Please download the pretained Glove Embedding from https://nlp.stanford.edu/projects/glove/')
            return
        pre_trained_weight = loadGloveModel(gloveFile, vocab_dic, len(vocab_dic) + 1)
    num_categories = 8
    train_data = ([], [])
    for data, label in doc_train_list:
        train_data[0].append(data)
        train_data[1].append(label)
    valid_data = ([], [])
    for data, label in doc_val_list:
        valid_data[0].append(data)
        valid_data[1].append(label)
    train_data = Data(train_data, max_num_sentence, num_categories, False)
    valid_data = Data(valid_data, max_num_sentence, num_categories, False)
    # model = DualOptim(args, graph_model, hypergraph_model, pre_trained_weight, len(vocab_dic) + 1).to(device)
    model = MLP(args, graph_model, hypergraph_model, pre_trained_weight, len(vocab_dic) + 1, num_categories).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    return train_data, valid_data, model, optimizer


# Shuffle the rows as it is arranged in time order -- the shuffle result is the same with the same random seed
#df = pd.read_csv("covid_dataset_shuffled.csv")
df = pd.read_csv("dataset/sbp_dataset.csv")
for label in labels:
    df[label] = df[label].fillna(0)
rounds, origin_idx = crossval(df, args.val_ratio)
data_labels = import_targets(df)
num_nodes = len(data_labels)
# G = nx.read_edgelist(args.dataset + "_" + args.feature + "_edgelist.gz")
# data_edge_index = [[int(edge[0]), int(edge[1])] for edge in G.edges]
# data_edge_index = np.array(data_edge_index).T
data_edge_index = np.arange(10)

HT = get_hypergraph(args.dataset, args.feature, num_nodes, args)
HT = np.array(HT)
print("* Hypergraph Size", HT.shape)

args.train = True
acc_categories = dict()
for label in labels:
    acc_categories[label] = []

graph_model = GraphConv(args).to(device)
hypergraph_model = HyperGAT(args).to(device)
# model = DualOptim(args, graph_model, hypergraph_model).to(device)
data_nodes, data_labels, HT, data_edge_index = numpy2tensor(data_nodes, data_labels, HT, data_edge_index)
if args.train:
    total_acc = []
    total_f1 = []
    for i, round in enumerate(rounds):
        graph_model = GraphConv(args).to(device)
        hypergraph_model = HyperGAT(args).to(device)
        # model = DualOptim(args, graph_model, hypergraph_model).to(device)
        model = MLP(args, graph_model, hypergraph_model).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
        train_idx = round[0]
        test_idx = round[1]
        # train_data, valid_data, model = import_data(train_idx, test_idx, origin_idx, df)
        origin_idx = np.arange(num_nodes)
        # train_data, test_data, model, optimizer = import_data(train_idx, test_idx, origin_idx)
        train_data, test_data, origin_data = split_data(train_idx, test_idx, origin_idx, data_labels)

        for epoch in range(args.epoch):
            loss = train(train_idx, data_edge_index)
            print("Epoch ", epoch, "loss: ", loss)

        acc, f1 = test(test_idx, data_edge_index, verbose=False, save=True)
        print("--Round ", i + 1, ": accuracy is", acc)
        total_acc.append(acc)
        total_f1.append(f1)
        '''
        for label in labels:
            acc_categories[label].append(acc_category[label])
        '''
    print("*" * 10)

    print("average accuracy: ", sum(total_acc) / len(total_acc), "std", statistics.stdev(total_acc))
    print("average f1: ", sum(total_f1) / len(total_acc), "std", statistics.stdev(total_f1))
    '''
    for label in labels:
        print(label, " category accuracy: ", sum(acc_categories[label]) / len(acc_categories[label]), "std",
              statistics.stdev(acc_categories[label]))
    '''
    state_dict = copy.deepcopy(model.state_dict())
    torch.save(state_dict, args.dataset + "_model.pt")

state_dict = torch.load(args.dataset + "_model.pt")
model.load_state_dict(state_dict)

args.attention = False
if args.attention:
    class_dict = cls_idx.class_idx(df)
    for pos, idx_list in class_dict.items():
        print("--", labels[pos])
        print(idx_list)
        print(len(idx_list))
        test_cls(idx_list, data_edge_index, labels[pos])
        print("*" * 10)
    import itertools
    for comb in itertools.combinations(labels, 2):
        index1 = comb[0]
        index2 = comb[1]
        class1 = labels.index(index1)
        class2 = labels.index(index2)
        print("Mixture of labels", index1, index2)

        cls1_idx_set = set(class_dict[class1])
        cls2_idx_set = set(class_dict[class2])
        idx_list = list(cls1_idx_set.intersection(cls2_idx_set))
        print(idx_list)
        test_cls(idx_list, data_edge_index, index1+"_"+index2)
        print("*" * 10)
