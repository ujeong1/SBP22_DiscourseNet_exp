#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import re
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

# In[1]:


import numpy as np
import pandas as pd
import re
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, confusion_matrix, multilabel_confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold

import warnings
warnings.filterwarnings('ignore')


# # Exploring Multi-label text classification with Machine Learning

# ## What is Multi-label classification
# 
# There are mainly tree types of classification problems:
#     
#     Bynary classification puts the samples in only one of two possible classes. A person can be either a male or a female.
# 
#     Multiclass classification means a classification task with more than two classes; e.g., classify a set of people in different age groups. Multiclass classification makes the assumption that each sample is assigned to one and only one label: a person can only be between 20 and 29 years old or 30 to 39 years old but not in both groups.
#     
#     Multilabel classification assigns to each sample a set of target labels. This can be thought as predicting properties of a data-point that are not mutually exclusive, such as topics that are relevant for a document. A text might be about any of religion, politics, finance or education at the same time or none of these.
# 
# There are two main methods for tackling a multi-label classification problem: problem transformation methods and algorithm adaptation methods.
# Problem transformation methods transform the multi-label problem into a set of binary classification problems, which can then be handled using single-class classifiers. They include OneVsRest and Binary Relevance techniques.
# 
# Whereas algorithm adaptation methods adapt the algorithms to directly perform multi-label classification. In other words, rather than trying to convert the problem to a simpler problem, they try to address the problem in its full form. They include Classifier Chains, Label Powerset and Adapted Algorithm.

# ## Our dataset
# 
# We will use the datasets from a Kaggle competition which aims to resolve the problem with negative online behaviors, like toxic comments (i.e. comments that are rude, disrespectful or otherwise likely to make someone leave a discussion). The dataset contains a large number of Wikipedia comments which have been labeled by human raters for toxic behavior. The types of toxicity are: **toxic, severe_toxic,obscene, threat, insult,identity_hate**. A model which predicts a probability of each type of toxicity for each comment must be created.
# 
# https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/overview
# 
# First we load our files:

# In[3]:


train_set = pd.read_csv('./input/jigsaw-toxic-comment-classification-challenge/train.csv')
test_set = pd.read_csv('./input/jigsaw-toxic-comment-classification-challenge/test.csv')
test_labels = pd.read_csv('./input/jigsaw-toxic-comment-classification-challenge/test_labels.csv')


# In[4]:


print('train set', train_set.head(10))
print('test set', test_set.head(10))
print('test labels', test_labels.head(10))


# ## Data cleaning and analisys
# 
# We need to check for empty values just in case:

# In[5]:


print(train_set.isna().sum())
print(test_set.isna().sum())
print(test_labels.isna().sum())


# As it was a competition, to deter hand labeling, the test_set contains some comments which are not included 
# in scoring. We need to remove the comments containing labels with -1 from the later supplied test_labels file and the test_set.

# In[28]:


test_set = test_set[test_labels['toxic'] != -1]
test_labels = test_labels[test_labels['toxic'] != -1]
test_features = test_set.comment_text
train_features = train_set.comment_text
train_labels = train_set.drop(['id', 'comment_text'], axis = 1)
test_labels = test_labels.drop(['id'], axis = 1)
labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

print(test_labels.head(10))
print(train_labels.head(10))


# We want to try to clean the text of the comments by removing punctuation and spcial characters, converting all to lowercase, stripping white space, and replacing some commonly used short forms with their full words. We want save the cleaned comments in separate variables and test the vectorizer algorithm with both sets. We will also use the stop_words parameter of the vectorizer algorithm to remove the words that do not bring value and will actually confuse and slow down the vectorizer.

# In[6]:


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

test_features_cleaned = test_features.map(lambda com : clean_text(com))
train_features_cleaned = train_features.map(lambda com : clean_text(com))


# We check the percentage of labeled and non-labeled comments

# In[7]:


print('Percentage of comments without labels: ')
print(len((train_set[(train_set.toxic == 0) & (train_set.severe_toxic == 0) & (train_set.obscene == 0) & (train_set.insult == 0) & (train_set.insult == 0) & (train_set.identity_hate == 0)])) / len(train_set)*100)
print('Percentage of comments with one or more labels: ')
print(len(train_set[(train_set.toxic == 1) | (train_set.severe_toxic == 1) | (train_set.obscene == 1) | (train_set.insult == 1) | (train_set.insult == 1) | (train_set.identity_hate == 1)]) / len(train_set)*100)


# In[8]:


test_labels.describe()


# Count the number of comments per label

# In[9]:


fig = plt.figure(figsize=(8,6))
ax = fig.add_axes([0,0,1,1])
total_count = []
for label in labels:
    total_count.append(len(train_labels[train_labels[label] == 1]))
ax.bar(labels,total_count, color=['red', 'green', 'blue', 'purple', 'orange', 'yellow'])
for i,data in enumerate(total_count):
    plt.text(i-.25, 
              data/total_count[i]+100, 
              total_count[i], 
              fontsize=12)
plt.title('Number of comments per label')
plt.xlabel('Labels')
plt.ylabel('Number of comments')

plt.show()


# For our purpose we are going to use OneVsRest classifier as the others are more complex and would take a considerable amount of time. OneVsRest classifier basically takes a classifier algorithm as a parameter and uses it to do a binary classification for each label in the dataset. We will also use pipelines to streamline the process by adding all estimators we need to be executed one after anoter. As the content we are going to analise is text we need first convert it into numeric values in order for our model to be able to process it. This is done by vectorizer algorithms that use different methods for comparing and weighing the relation between words in a single document and their relations in the whole corpus. They first collect all the unique words in a document(comment in our case) and build a dictionary of them and then vectorise each one giving it a specific weigth based on how much a word relates to a specific label within the whole corpus(the collection of all comments in our dataset). We are going to use TfidfVectorizer which is one of the widely used vectorisers.

# ## Creating baseline models
# 
# In order to choose our best classifier and improve it further we are going to test tree different classifiers - **NaiveBayes, LogisticRegression and SVM with linear kernel** using their default parameters. 
#     
# The еvaluation metrics we are going to use to judge our models :
#     
#     As specified in the competition requirements we need to use AUC. 
#     
#     As suggested in some other articles we will also use micro-averaging for all labels of precision, recall and F1-score
#     
# LinearSVM algorithm does not have a predict_proba method which makes it impossible to use roc_auc_score for this model. I tried with SVC(kernel='linear',probability=True) which should make it possible but it took forever to compute so I just had to abandon it. We will just use acuracy_score and micro_averages F1-score.

# In[10]:


NB_pipeline = Pipeline([('tfidf', TfidfVectorizer(stop_words='english')),
                       ('nb_model', OneVsRestClassifier(MultinomialNB(), n_jobs=-1))])

LR_pipeline = Pipeline([('tfidf', TfidfVectorizer(stop_words='english')),
                       ('lr_model', OneVsRestClassifier(LogisticRegression(), n_jobs=-1))])

# SVM_pipeline = Pipeline([('tfidf', TfidfVectorizer(stop_words='english')),
#                        ('svm_model', OneVsRestClassifier(SVC(kernel='linear',probability=True)))])

SVM_pipeline = Pipeline([('tfidf', TfidfVectorizer(stop_words='english')),
                       ('svm_model', OneVsRestClassifier(LinearSVC(), n_jobs=-1))])

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
    print('roc_auc: ', roc_auc_score(test_lbls, pred_proba))
    print('accuracy: ', accuracy_score(test_lbls, predictions))
    print('confusion matrices: ')
    print(multilabel_confusion_matrix(test_lbls, predictions))
    print('classification_report: ')
    print(classification_report(test_lbls, predictions, target_names=labels))
    
def run_SVM_pipeline(pipeline, train_feats, train_lbls, test_feats, test_lbls):
    pipeline.fit(train_feats, train_labels)
    predictions = pipeline.predict(test_feats)
    print('accuracy: ', accuracy_score(test_lbls, predictions))
    print('confusion matrices: ')
    print(multilabel_confusion_matrix(test_lbls, predictions))
    print('classification_report: ')
    print(classification_report(test_lbls, predictions, target_names=labels))
    
def plot_pipeline_roc_curve(pipeline, train_feats, train_lbls, test_feats, test_lbls):
    for label in labels:
        pipeline.fit(train_feats, train_set[label])
        pred_proba = pipeline.predict_proba(test_feats)[:,1]
        plot_roc_curve(test_lbls[label], pred_proba)


# In[11]:


# run_pipeline(NB_pipeline, train_features, train_labels, test_features, test_labels)


# In[26]:


# run_pipeline(SVM_pipeline, train_features, train_labels, test_features, test_labels)
train_labels.rename(columns = {'severe_toxic':'severe toxis'}, inplace=True)
run_SVM_pipeline(SVM_pipeline, train_features, train_labels, test_features, test_labels)


# In[19]:

exit(0)
pd.set_option('display.max_rows', None)

print(test_labels[:100])


# Plot ROC curve for LogisticRegression model

# In[ ]:


plot_pipeline_roc_curve(LR_pipeline, train_features, train_labels, test_features, test_labels)


# Plot ROC curve for NaiveBayes model

# In[ ]:


plot_pipeline_roc_curve(NB_pipeline, train_features, train_labels, test_features, test_labels)


# ## Hyperparameter tunning
# 
# We select the LogisticRegression classifier as our best and will try to improve our results with the following techniques:
#     improve the tf-idf vectorizer's parameters
#     improve the tf-idf vectorizer by cleaning the text of the comments
#     improve the logistic regression's parameters
#     
# We checked the distribution of samples in our dataset earlier and we saw that it was highly unbalanced - 90% features with no labels and 10% features with one or more labels.
# In this situation in case of bynary or multiclass classification we would be able to use GridSearch with StratifiedKFold cross validation splits to test a set of different parameters for every algorithm in our pipeline. In our case though, the StratifiedKFold does not suppport multi-label and using just KFold would not yeld any useful data splits. I decided to give it a try anyway to prove this or burst it. 

# In[ ]:


# TAKES ABOUT 40min to run
# alpha = [0.1,1,10]
# penalty=['l1','l2']
# n_gram=[(1,1),(1,2)]
# param_grid = {
#     'tfidf__ngram_range': n_gram,
#     'lr_model__estimator__C': alpha
# }
# gsearch_cv = GridSearchCV(LR_pipeline, param_grid=param_grid, cv=5)
# gsearch_cv.fit(train_features, train_labels)


# In[ ]:


gsearch_cv.best_score_
# 0.919784923325667


# In[ ]:


gsearch_cv.best_params_
# {'lr_model__estimator__C': 10, 'tfidf__ngram_range': (1, 2)}


# In[ ]:


LR_pipeline = Pipeline([('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2))),
                       ('lr_model', OneVsRestClassifier(LogisticRegression(C=10), n_jobs=-1))])


# In[ ]:


run_pipeline(LR_pipeline, train_features, train_labels, test_features, test_labels)


# Running the pipeline with the suggested best parameters actually resulted in getting a better AUC accuracy. My asumption is that it was just by chance as there is no way to do a proper training on such an inbalanced set without stratifying the folds. We will proceed with a manual tunning of the models' parameters anyway to make a comparison.

# **We try the cleaned comments with the default parameters used for our baseline best model and see almost no difference, a slight decrease in fact with 0.0001:**
# 
# with cleaned comments and C=1, ngram_range=(1,1) - roc_auc:  0.9750820030803234
# micro avg       0.69      0.59      0.63     14498
# 
# **Second thing we try to increase n-gram range of the tf-idf. n-gram parameter defines the number of consecutive words to be combined when creating the vectorising the text content. (1,1) is the default value which means only single words will be used**
# 
# with not cleaned comments and C=1, ngram_range=(1,2) - roc_auc:  0.9734608599958284 
# micro avg       0.68      0.59      0.63     14498
# 
# with not cleaned comments and C=1, ngram_range=(2,2) - roc_auc:  0.8695484960212974
# micro avg       0.86      0.08      0.15     14498
# 
# with not cleaned comments and C=1, ngram_range=(1,3) - roc_auc:  0.9721529511622968
# micro avg       0.66      0.61      0.63     14498
# 
# **different values for LogisticRegression parameters
# start with C wich is the regularization strength parameter - smaller values specify stronger regularization.**
# 
# with not cleaned comments and C=.5, ngram_range=(1,1) - roc_auc:  0.9739737234377395
# micro avg       0.73      0.53      0.61     14498
# 
# with not cleaned comments and C=2, ngram_range=(1,1) - roc_auc:  0.9752981961370573
# micro avg       0.67      0.63      0.64     14498
# 
# with not cleaned comments and C=3, ngram_range=(1,1) - roc_auc:  0.9749554220049567
# micro avg       0.65      0.64      0.65     14498
# 
# with not cleaned comments and C=2.1, ngram_range=(1,1) - roc_auc:  0.9752756613723531
# micro avg       0.66      0.63      0.65     14498
# 
# with not cleaned comments and C=1.9, ngram_range=(1,1) - roc_auc:  0.9753144853229717
# micro avg       0.67      0.62      0.64     14498
# 
# **OUR BEST PARAMS**
# **with cleaned comments and C=1.5, ngram_range=(1,1) - roc_auc:  0.9753254734889681
# micro avg       0.67      0.62      0.64     14498**
# 
# with not cleaned comments and C=1.4, ngram_range=(1,1) - roc_auc:  0.9753073320518185
# micro avg       0.67      0.62      0.64     14498
# 
# **We select C=1.5 to be the best value and then we lastly try the penalty parameter of 'l1' as the default is 'l2'**
# 
# with not cleaned comments and C=1.5, l1, ngram_range=(1,1) - roc_auc:  0.9725936377263187
# micro avg       0.67      0.62      0.64     14498

# In[22]:


# change parameters manually to get the results above
LR_pipeline = Pipeline([('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1,2))),
                       ('lr_model', OneVsRestClassifier(LogisticRegression(C=10), n_jobs=-1))])


# In[23]:


run_pipeline(LR_pipeline, train_features_cleaned, train_labels, test_features_cleaned, test_labels)


# ## Conclusion
# 
# Using relatively simple methods and algorithms we managed to create a model with an accuracy above 97% which is quite high.
# We could not 100% prove that cross-validation is not applicable for multi-label classification problems. Probably we need to run GridSearchCV few more times to see if we will get always tha same results.
# Using the created setup we can continue trying to improve our model by:
# 
# 1. Performing different types of stemming of the words
# 2. Try other algorithms for vectorizing comments' content like word2vec
# 3. Try using character n-grams instead of word n-grams
# 4. Try using ensemble classification algorithms

# ## RESOURCES
# 
# 1. https://en.wikipedia.org/wiki/Multi-label_classification
# 
# 2. https://scikit-learn.org/stable/modules/multiclass.html
# 
# 3. https://medium.com/@saugata.paul1010/a-detailed-case-study-on-multi-label-classification-with-machine-learning-algorithms-and-72031742c9aa
# 
# 4. https://towardsdatascience.com/multi-label-text-classification-with-scikit-learn-30714b7819c5
# 
# 5. https://www.analyticsvidhya.com/blog/2017/08/introduction-to-multi-label-classification/
# 
