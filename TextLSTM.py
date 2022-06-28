import argparse
import statistics
from baseline_utils.utils import split_validation, Data
from baseline_utils.preprocess import *
from model import *
import random
import warnings
import os
from sklearn.metrics import accuracy_score
import pandas as pd
import torch.nn.functional as F
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix, classification_report
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='covid', help='dataset name')
parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=128, help='hidden state size')
parser.add_argument('--initialFeatureSize', type=int, default=300, help='initial size')
parser.add_argument('--epoch', type=int, default=40, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-6, help='l2 penalty')
parser.add_argument('--valid_portion', type=float, default=0.1,
                    help='split the portion of training set as validation set')
parser.add_argument('--rand', type=int, default=1234, help='rand_seed')
parser.add_argument('--normalization', action='store_true', help='add a normalization layer to the end')
parser.add_argument('--use_LDA', action='store_true', help='use LDA to construct semantic hyperedge')

args = parser.parse_args()
print(args)

SEED = args.rand
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

labels_dict = {"Prevention": 0, "Treatment": 1, "Diagnosis": 2, "Mechanism": 3, "Case Report": 4, "Transmission": 5,
               "Forecasting": 6,
               "General": 7}
labels = list(labels_dict.keys())


class TextLSTM(nn.Module):
    def __init__(self, opt, pre_trained_weight, n_node, n_categories):
        super().__init__()
        self.n_node = n_node
        self.hidden_size = 100
        self.initial_feature = 300
        self.n_categories = n_categories
        self.embedding = nn.Embedding(self.n_node + 1, self.initial_feature, padding_idx=0)
        self.pretrain = opt.pretrained
        if self.pretrain:
            pre_trained_weight = torch.FloatTensor(pre_trained_weight)
            self.embedding = nn.Embedding.from_pretrained(pre_trained_weight, freeze = False, padding_idx = 0)
        self.bidirection = False
        if self.bidirection:
            self.prediction_transform = nn.Linear(self.hidden_size*2, self.n_categories, bias=True)
        else:
            self.prediction_transform = nn.Linear(self.hidden_size, self.n_categories, bias=True)
        layer_dim = 1
        self.lstm = nn.LSTM(self.initial_feature,self.hidden_size, layer_dim, batch_first=True, bidirectional=self.bidirection)

        self.reset_parameters()
        d_prob = 0.5
        self.dropout = nn.Dropout(d_prob)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)

    def forward(self, items):
        x = self.embedding(items)
        # x = x.transpose(1,2)
        x, _ = self.lstm(x)
        # outputs = self.pool(outputs)
        return x[:, -1, :]

    def compute_scores(self, x, mask):
        result = self.prediction_transform(x)
        # x = self.layer_normH(x)
        return result

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


def crossval(ratio, df):
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

    # Does the origin idx should be in ascending in order?
    origin_idx = np.arange(num_nodes)
    # np.random.shuffle(origin_idx)

    rounds = []
    num_val = int(num_nodes * ratio)
    steps = 5
    for step in range(steps):
        start = int((step * ratio) * num_nodes)
        end = min(int(start + num_val), num_nodes)+1
        val_idx = origin_idx[start:end].tolist()
        train_idx = origin_idx.copy()
        train_idx = np.delete(train_idx, np.arange(start, end), None)
        np.random.shuffle(train_idx)
        train_idx = train_idx.tolist()
        rounds.append((train_idx, val_idx))
    return rounds, origin_idx


def import_data(train_idx, val_idx, origin_idx, df):
    doc_content_list, doc_train_list, doc_val_list, vocab_dic, max_num_sentence = read_file(args.dataset, train_idx,
                                                                                            val_idx, origin_idx, df)
    pre_trained_weight = []
    if args.pretrained:
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
    train_data = Data(train_data, max_num_sentence, num_categories, args.use_LDA)
    valid_data = Data(valid_data, max_num_sentence, num_categories, args.use_LDA)
    model = trans_to_cuda(TextLSTM(args, pre_trained_weight, len(vocab_dic) + 1, num_categories))
    return train_data, valid_data, model


def main():
    args.pretrained=False
    total_acc = []
    total_f1 = []
    acc_categories = dict()
    for label in labels:
        acc_categories[label] = []
    df = pd.read_csv("dataset/sbp_dataset.csv")#pd.read_csv(args.dataset+"_dataset_shuffled.csv")
    ratio = 0.2
    rounds, origin_idx = crossval(ratio, df)
    for round in rounds:
        train_idx = round[0]
        valid_idx = round[1]
        train_data, valid_data, model = import_data(train_idx, valid_idx, origin_idx, df)
        for epoch in range(args.epoch):
            print('-------------------------------------------------------')
            print('epoch: ', epoch)

            train_cnn(model, train_data, args)

        acc, f1 = test_cnn(model, valid_data, args, False)

        total_acc.append(acc)
        total_f1.append(f1)
        '''
        for label in labels:
            acc_categories[label].append(acc_category[label])
        '''
        print(acc, f1)
    print(total_acc)
    avg_acc = sum(total_acc)/len(total_acc)
    print("average acc, ", avg_acc)
    print("std, ", statistics.stdev(total_acc))
    avg_f1 = sum(total_f1)/len(total_f1)
    print("average acc, ", avg_f1)
    print("std, ", statistics.stdev(total_f1))
    '''
    for label in labels:
        print(label, " category accuracy: ", sum(acc_categories[label]) / len(acc_categories[label]), "std", statistics.stdev(acc_categories[label]))
    '''
if __name__ == '__main__':
        main()
