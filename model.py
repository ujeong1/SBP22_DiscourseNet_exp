import datetime
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
from sklearn import metrics
from layers import *
from tqdm import tqdm
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, global_max_pool
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix, classification_report

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


class DualOptim(nn.Module):
    def __init__(self, args, graph_model, hypergraph_model):
        super().__init__()
        self.gm = graph_model
        self.hm = hypergraph_model
        self.hidden_channels = args.hidden_channels
        self.out_channels = args.num_classes
        self.in_channels = args.in_channels
        # self.pretrained=False
        # self.n_node = n_node
        # self.embedding = nn.Embedding(self.n_node + 1, self.in_channels, padding_idx=0)
        # if self.pretrained:
        #     pre_trained_weight = torch.FloatTensor(pre_trained_weight)
        #     self.embedding = nn.Embedding.from_pretrained(pre_trained_weight, freeze = False, padding_idx = 0)
        self.lin = nn.Linear(self.hidden_channels * 2, self.hidden_channels)
        self.lin1 = nn.Linear(self.in_channels, 2 * self.hidden_channels)
        self.lin2 = nn.Linear(2 * self.hidden_channels, self.hidden_channels)
        self.cls = nn.Linear(self.hidden_channels*2, self.out_channels)
        self.normalization = False
        self.layer_normC = nn.LayerNorm(self.hidden_channels*2, eps=1e-6)
        self.dropout = nn.Dropout(args.dropout)
        self.reset_parameters()

    def forward(self, x, A, H):
        #x = self.dropout(x)
        # g = self.gm(x, A)

        # x = self.lin1(x)
        # x = self.embedding(x)
        g = self.lin1(x)#.relu()

        h, edge = self.hm(g.unsqueeze(0), H)

        γ = self.lin2(g)#.relu()
        h = h + γ
        #h = self.dropout(h)

        return self.cls(h.squeeze(0)), edge

    def compute_score(self, x):
        # x = self.layer_normH(x)
        if self.normalization:
            x = self.layer_normC(x)
        pred = self.cls(x)
        return pred  # F.log_softmax(v, dim=-1)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_channels)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


class MLP(nn.Module):
    def __init__(self, args, graph_model, hypergraph_model):
        super().__init__()
        self.gm = graph_model
        self.hm = hypergraph_model
        self.hidden_channels = args.hidden_channels
        self.out_channels = args.num_classes
        self.in_channels = args.in_channels
        #
        # self.n_categories = n_categories
        # self.embedding = nn.Embedding(self.n_node + 1, self.in_channels, padding_idx=0)

        self.lin = nn.Linear(self.hidden_channels * 2, self.hidden_channels)
        self.lin1 = nn.Linear(self.in_channels, 2 * self.hidden_channels)
        self.lin2 = nn.Linear(2 * self.hidden_channels, self.hidden_channels)
        self.cls = nn.Linear(self.hidden_channels, self.out_channels)
        self.normalization = False
        self.layer_normH = nn.LayerNorm(self.hidden_channels, eps=1e-6)
        self.reset_parameters()

    def forward(self, x, A, H):
        # x = self.embedding(x)
        h = self.lin1(x)  # g+x
        h = self.lin2(h)

        return self.cls(h.squeeze(0)), h

    def compute_score(self, x):
        x = self.layer_normH(x)
        pred = self.cls(x)
        if self.normalization:
            pred = self.layer_normC(pred)
        return pred  # F.log_softmax(v, dim=-1)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_channels)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


class GraphConv(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.in_channels = args.in_channels
        self.hidden_channels = args.hidden_channels * 2
        self.out_channels = args.num_classes
        self.model = args.model
        if self.model == 'GCN':
            self.conv = GCNConv(self.in_channels, self.hidden_channels)
        elif self.model == 'SAGE':
            self.conv = SAGEConv(self.in_channels, self.hidden_channels)
        elif self.model == 'GAT':
            self.conv = GATConv(self.in_channels, self.hidden_channels)
        # self.cls= nn.Linear(self.hidden_channels, self.out_channels)

    def forward(self, x, edge_index):
        g = self.conv(x, edge_index)
        return g

    def compute_score(self, x):
        # v = self.cls(g)
        # return F.log_softmax(v, dim=-1)
        pass

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_channels)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


class HyperConv(nn.Module):
    def __init__(self, args, bias=True):
        super().__init__()
        self.in_channels = args.hidden_channels * 2
        self.hidden_channels = args.hidden_channels
        self.out_channels = args.num_classes
        self.model = args.model
        self.weight = Parameter(torch.Tensor(self.in_channels, self.hidden_channels))
        if bias:
            self.bias = Parameter(torch.Tensor(self.hidden_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x)
        return x

    def compute_score(self, x):
        # v = self.lin(g)
        # return F.log_softmax(v, dim=-1)
        pass

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_channels)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


class HGNN_ATT(nn.Module):
    def __init__(self, input_size, n_hid, output_size, dropout=0.3):
        super(HGNN_ATT, self).__init__()
        self.dropout = dropout
        self.gat1 = HyperGraphAttentionLayerSparse(input_size, n_hid, dropout=self.dropout, alpha=0.2, transfer=False,
                                                   concat=True)
        self.gat2 = HyperGraphAttentionLayerSparse(n_hid, output_size, dropout=self.dropout, alpha=0.2, transfer=True,
                                                   concat=False)

    def forward(self, x, H):
        x, edge = self.gat1(x, H)
        # x = F.dropout(x, self.dropout, training=self.training)
        # x, edge = self.gat2(x, H)
        return x, edge


class HyperGAT(Module):
    def __init__(self, args):
        super(HyperGAT, self).__init__()
        self.initial_feature = args.hidden_channels * 2
        self.hidden_size = args.hidden_channels
        self.dropout = args.dropout

        self.hgnn = HGNN_ATT(self.initial_feature, self.initial_feature, self.hidden_size, dropout=self.dropout)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, v, HT):
        h, edge = self.hgnn(v, HT)  # documents are nodes and inputs
        return h, edge

    def compute_score(self, x):
        # v = self.lin(g)
        # return F.log_softmax(v, dim=-1)
        pass


class DocumentGraph(Module):
    def __init__(self, opt, pre_trained_weight, n_node, n_categories):
        super(DocumentGraph, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.n_categories = n_categories
        self.batch_size = opt.batchSize
        self.dropout = opt.dropout
        self.initial_feature = opt.initialFeatureSize
        self.normalization = opt.normalization
        self.dataset = opt.dataset
        self.pretrained = opt.pretrained
        self.embedding = nn.Embedding(self.n_node + 1, self.initial_feature, padding_idx=0)
        if self.pretrained:
            pre_trained_weight = torch.FloatTensor(pre_trained_weight)
            self.embedding = nn.Embedding.from_pretrained(pre_trained_weight, freeze=False, padding_idx=0)
        self.layer_normH = nn.LayerNorm(self.hidden_size, eps=1e-6)
        if self.normalization:
            self.layer_normC = nn.LayerNorm(self.n_categories, eps=1e-6)

        self.prediction_transform = nn.Linear(self.hidden_size, self.n_categories, bias=True)

        self.reset_parameters()

        self.hgnn = HGNN_ATT(self.initial_feature, self.initial_feature, self.hidden_size, dropout=self.dropout)

        # self.class_weights = class_weights
        # self.loss_function = nn.CrossEntropyLoss(weight = trans_to_cuda(torch.Tensor(self.class_weights).float()))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_scores(self, inputs, node_masks):

        hidden = inputs * node_masks.view(node_masks.shape[0], -1, 1).float()
        b = torch.sum(hidden * node_masks.view(node_masks.shape[0], -1, 1).float(), -2) / torch.sum(node_masks,
                                                                                                    -1).repeat(
            hidden.shape[2], 1).transpose(0, 1)
        #b = self.layer_normH(b)
        b = self.prediction_transform(b)

        pred = b

        if self.normalization:
            pred = self.layer_normC(b)

        return pred

    def forward(self, inputs, HT):

        hidden = self.embedding(inputs)
        # _ is for edge output
        nodes, _ = self.hgnn(hidden, HT)
        return nodes


def forward2(model, alias_inputs, HT, items, targets, node_masks):
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
    items = trans_to_cuda(torch.Tensor(items).long())
    HT = trans_to_cuda(torch.Tensor(HT).float())
    node_masks = trans_to_cuda(torch.Tensor(node_masks).float())
    node = model(items, HT)
    get = lambda i: node[i][alias_inputs[i]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
    targets = trans_to_cuda(torch.Tensor(targets).float())
    return targets, model.compute_scores(seq_hidden, node_masks)
def forward(model, alias_inputs, HT, items, targets, node_masks):
    #alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
    items = trans_to_cuda(torch.Tensor(items).long())
    #HT = trans_to_cuda(torch.Tensor(HT).float())
    node_masks = trans_to_cuda(torch.Tensor(node_masks).float())
    outputs = model(items)
    # get = lambda i: node[i][alias_inputs[i]]
    # seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
    targets = trans_to_cuda(torch.Tensor(targets).float())
    return targets, model.compute_scores(outputs, node_masks)


def loss_fn(outputs, targets):
    return nn.BCEWithLogitsLoss()(outputs, targets)


def train_cnn(model, train_data, opt):
    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(opt.batchSize, True)
    idx = 0
    for step in tqdm(range(len(slices)), total=len(slices), ncols=70, leave=False, unit='b'):
        i = slices[step]
        model.optimizer.zero_grad()
        alias_inputs, HT, items, targets, node_masks = train_data.get_slice(i)
        targets, outputs = forward(model, alias_inputs, HT, items, targets, node_masks)
        loss = loss_fn(outputs, targets)
        loss.backward()
        model.optimizer.step()
        total_loss = total_loss + (1 / (idx + 1) * (loss.item() - total_loss))
        idx += 1

    print('\tLoss:\t%.4f' % (total_loss))


def train_model(model, train_data, opt):
    # model.scheduler.step()
    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(opt.batchSize, True)
    idx = 0
    for step in tqdm(range(len(slices)), total=len(slices), ncols=70, leave=False, unit='b'):
        i = slices[step]
        alias_inputs, HT, items, targets, node_masks = train_data.get_slice(i)
        model.optimizer.zero_grad()
        targets, outputs = forward2(model, alias_inputs, HT, items, targets, node_masks)
        loss = loss_fn(outputs, targets)
        #targets = trans_to_cuda(torch.Tensor(targets).float())
        #loss = loss_fn(scores, targets)
        loss.backward()
        model.optimizer.step()
        total_loss = total_loss + (1 / (idx + 1) * (loss.item() - total_loss))
        idx += 1

    print('\tLoss:\t%.4f' % (total_loss))


def test_cnn(model, test_data, opt, verbose=True):
    model.eval()

    test_pred = []
    test_labels = []
    slices = test_data.generate_batch(10, False)
    num_samples = test_data.length
    for step in tqdm(range(len(slices)), total=len(slices), ncols=70, leave=False, unit='b'):
        i = slices[step]
        alias_inputs, HT, items, targets, node_masks = test_data.get_slice(i)
        targets, outputs = forward(model, alias_inputs, HT, items, targets, node_masks)
        scores = torch.sigmoid(outputs)
        preds = np.round(scores.cpu().detach().numpy())
        test_labels += list(targets.cpu().detach().numpy())
        test_pred += list(preds)
    acc = accuracy_score(test_pred, test_labels)
    labels = ["Prevention", "Treatment", "Diagnosis", "Mechanism", "Case Report", "Transmission", "Forecasting",
              "General"]
    acc_category = dict()
    for label in labels:
        acc_category[label] = 0
    for i in range(len(targets)):
        for j, label in enumerate(labels):
            if test_pred[i][j] == targets[i][j]:
                acc_category[label] += 1
    #print(test_pred)
    for label in labels:
        acc_category[label] /= num_samples
    preds = test_pred
    targets = test_labels
    report = classification_report(preds, targets, target_names=labels, output_dict=True)
    f1 = report['macro avg']['f1-score']
    acc = report['micro avg']['f1-score']

    return acc, f1


def test_model(model, test_data, opt, verbose=True):
    model.eval()

    test_pred = []
    test_labels = []
    slices = test_data.generate_batch(10, False)
    num_samples = test_data.length
    for step in tqdm(range(len(slices)), total=len(slices), ncols=70, leave=False, unit='b'):
        i = slices[step]
        alias_inputs, HT, items, targets, node_masks = test_data.get_slice(i)
        targets, scores = forward2(model, alias_inputs, HT, items, targets, node_masks)
        scores = torch.sigmoid(scores)
        outputs = np.round(scores.cpu().detach().numpy())
        test_labels += list(targets.cpu().detach().numpy())
        test_pred += list(outputs)
        print(outputs)

    acc = accuracy_score(test_pred, test_labels)
    labels = ["Prevention", "Treatment", "Diagnosis", "Mechanism", "Case Report", "Transmission", "Forecasting",
              "General"]
    acc_category = dict()
    for label in labels:
        acc_category[label] = 0
    for i in range(len(targets)):
        for j, label in enumerate(labels):
            if test_pred[i][j] == targets[i][j]:
                acc_category[label] += 1

    for label in labels:
        acc_category[label] /= num_samples
    preds = test_pred
    targets = test_labels
    report = classification_report(preds, targets, target_names=labels, output_dict=True)
    f1 = report['macro avg']['f1-score']
    return acc, f1
