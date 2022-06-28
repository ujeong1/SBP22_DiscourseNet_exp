import numpy as np
import pandas as pd
import re
import numpy as np
import pandas as pd
import re
import statistics
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, confusion_matrix, \
multilabel_confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold


labels = ["Prevention", "Treatment", "Diagnosis", "Mechanism", "Case Report", "Transmission", "Forecasting", "General"]

label_list = ["Prevention", "Treatment", "Diagnosis", "Mechanism", "Case Report", "Transmission", "Forecasting",
              "General"]


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
        end = min(int(start + num_val), num_nodes)
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


def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    return text


def plot_roc_curve(test_features, predict_prob):
    fpr, tpr, thresholds = roc_curve(test_features, predict_prob)
    plt.plot(fpr, tpr)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.title('ROC curve for toxic comments')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(True)
    plt.legend(labels)


def run_pipeline(pipeline, train_feats, train_lbls, test_feats, test_lbls):
    pipeline.fit(train_feats, train_labels)
    predictions = pipeline.predict(test_feats)
    pred_proba = pipeline.predict_proba(test_feats)
    # print('roc_auc: ', roc_auc_score(test_lbls, pred_proba))
    print('accuracy: ', accuracy_score(test_lbls, predictions))
    # print('confusion matrices: ')
    # print(multilabel_confusion_matrix(test_lbls, predictions))
    print('classification_report: ')
    print(classification_report(test_lbls, predictions, target_names=labels))
    targets = list(test_lbls.values)
    preds = predictions
    acc_category = dict()
    for label in label_list:
        acc_category[label] = 0
    for i in range(len(targets)):
        for j, label in enumerate(label_list):
            if preds[i][j] == targets[i][j]:
                acc_category[label] += 1
    num_samples = len(targets)
    for label in labels:
        acc_category[label] /= num_samples
    # print(acc_category)
    # print('confusion matrices: ')
    # print(multilabel_confusion_matrix(test_lbls, predictions))
    print('classification_report: ')
    print(classification_report(test_lbls, predictions, target_names=labels))
    return accuracy_score(test_lbls, predictions), acc_category


def run_SVM_pipeline(pipeline, train_feats, train_lbls, test_feats, test_lbls):
    # print(test_lbls)
    # print(train_feats)
    pipeline.fit(train_feats, train_lbls)
    predictions = pipeline.predict(test_feats)
    print('accuracy: ', accuracy_score(test_lbls, predictions))
    targets = list(test_lbls.values)
    preds = predictions
    acc_category = dict()
    for label in label_list:
        acc_category[label] = 0
    for i in range(len(targets)):
        for j, label in enumerate(label_list):
            if preds[i][j] == targets[i][j]:
                acc_category[label] += 1
    num_samples = len(targets)
    for label in labels:
        acc_category[label] /= num_samples
    # print(acc_category)
    # print('confusion matrices: ')
    # print(multilabel_confusion_matrix(test_lbls, predictions))
    print('classification_report: ')
    print(classification_report(test_lbls, predictions, target_names=labels))
    return accuracy_score(test_lbls, predictions), acc_category


def plot_pipeline_roc_curve(pipeline, train_feats, train_lbls, test_feats, test_lbls):
    for label in labels:
        pipeline.fit(train_feats, train_set[label])
        pred_proba = pipeline.predict_proba(test_feats)[:, 1]
        plot_roc_curve(test_lbls[label], pred_proba)


import warnings

warnings.filterwarnings('ignore')
origin_set = pd.read_csv('dataset/sbp_dataset.csv')
df = origin_set
rounds, origin_idx = crossval(0.2, df)
data_labels = import_targets(df)
num_nodes = len(data_labels)
total = []
acc_categories = dict()
for label in label_list:
    acc_categories[label] = []
for round in rounds:
    train_idx = round[0]
    test_idx = round[1]
    train_set = origin_set.iloc[train_idx]
    test_set = origin_set.iloc[test_idx]
    train_labels =pd.DataFrame()
    for label in labels:
        train_labels[label] = train_set[label]
    test_labels =pd.DataFrame()
    for label in labels:
        test_labels[label] = test_set[label]

    test_features = test_set.ad_creative_body#.processed_text
    train_features = train_set.ad_creative_body#.processed_text

    test_labels = test_labels.fillna(0).astype(int)
    train_labels = train_labels.fillna(0).astype(int)

    test_features_cleaned = test_features.map(lambda com: clean_text(com))
    train_features_cleaned = train_features.map(lambda com: clean_text(com))

    SVM_pipeline = Pipeline([('tfidf', TfidfVectorizer(stop_words='english')),
                             ('svm_model', OneVsRestClassifier(LinearSVC(), n_jobs=-1))])

    LR_pipeline = Pipeline([('tfidf', TfidfVectorizer(stop_words='english')),
                           ('lr_model', OneVsRestClassifier(LogisticRegression(), n_jobs=-1))])
    #acc, acc_category = run_pipeline(LR_pipeline, train_features, train_labels, test_features, test_labels)
    acc, acc_category = run_SVM_pipeline(SVM_pipeline, train_features, train_labels, test_features, test_labels)#not cleaned?
    total.append(acc)
    for label in label_list:
        acc_categories[label].append(acc_category[label])
print("avg acc", sum(total)/5, statistics.stdev(total))
for label in label_list:
    print(label, " category accuracy: ", sum(acc_categories[label]) / len(acc_categories[label]), "std",
          statistics.stdev(acc_categories[label]))
