from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
import pandas as pd
from time import time
import nltk
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import sklearn.metrics
from pandas import DataFrame,Series
import matplotlib.pyplot as plt

training_time_container={'b_naive_bayes':0,'mn_naive_bayes':0,'random_forest':0,'linear_svm':0}
prediction_time_container={'b_naive_bayes':0,'mn_naive_bayes':0,'random_forest':0,'linear_svm':0}
accuracy_container={'b_naive_bayes':0,'mn_naive_bayes':0,'random_forest':0,'linear_svm':0}


################################################
variables = dataframe_x.iloc[:,:].values
labels = dataframe_y.iloc[:,:].values
labels = labels.astype('float64')


################################################
from sklearn.naive_bayes import BernoulliNB
bnb_classifier=BernoulliNB()
t0=time()
bnb_classifier=bnb_classifier.fit(variables,labels)
training_time_container['b_naive_bayes']=time()-t0

t0=time()
bnb_predictions=bnb_classifier.predict(variables)
prediction_time_container['b_naive_bayes']=time()-t0
prediction_time_container['b_naive_bayes']

nb_ascore=sklearn.metrics.accuracy_score(labels, bnb_predictions)
accuracy_container['b_naive_bayes']=nb_ascore

print("Bernoulli Naive Bayes Accuracy Score: %f"%accuracy_container['b_naive_bayes'])
print("Training Time: %f"%training_time_container['b_naive_bayes'])
print("Prediction Time: %f"%prediction_time_container['b_naive_bayes'])

from sklearn.metrics import f1_score
bayes_f1_score = f1_score(labels, bnb_predictions, average='macro')
print("Bernoulli Naive Bayes F1 Score:",bayes_f1_score)

from sklearn.metrics import precision_score
bayes_precision_score = precision_score(labels, bnb_predictions, average='macro')
print("Bernoulli Naive Precision Score:",bayes_precision_score)

from sklearn.metrics import recall_score
bayes_recall_score = recall_score(labels, bnb_predictions, average='macro')
print("Bernoulli Naive Bayes Recall Score:",bayes_recall_score)

#################################################################

variables_train, variables_test, labels_train, labels_test=train_test_split(
        variables, labels, test_size=.9)

bnb_classifier=BernoulliNB()
bnb_classifier=bnb_classifier.fit(variables_train,labels_train)
bnb_predictions=bnb_classifier.predict(variables_test)
nb_ascore=sklearn.metrics.accuracy_score(labels_test, bnb_predictions)
print(nb_ascore)

from sklearn.metrics import f1_score
bayes_f1_score = f1_score(labels_test, bnb_predictions, average='macro')
print("Bernoulli Naive Bayes F1 Score:",bayes_f1_score)

from sklearn.metrics import precision_score
bayes_precision_score = precision_score(labels_test, bnb_predictions, average='macro')
print("Bernoulli Naive Precision Score:",bayes_precision_score)

from sklearn.metrics import recall_score
bayes_recall_score = recall_score(labels_test, bnb_predictions, average='macro')
print("Bernoulli Naive Bayes Recall Score:",bayes_recall_score)

################################################################

fold_validation_sum = 0
fold_validation_f1 = 0
fold_validation_precision = 0
fold_validation_recall = 0

from sklearn.model_selection import KFold, cross_val_score
k_fold = KFold(n_splits=10,shuffle=True)
for train_indices, test_indices in k_fold.split(variables):
    x_train, x_test = variables[train_indices], variables[test_indices]
    y_train, y_test = labels[train_indices], labels[test_indices]
    
    bnb_classifier=BernoulliNB()
    bnb_classifier=bnb_classifier.fit(x_train,y_train)
    pred=bnb_classifier.predict(x_test)

    accuracy=sklearn.metrics.accuracy_score(y_test, pred)
    fold_validation_sum = fold_validation_sum + accuracy
    
    from sklearn.metrics import f1_score
    bayes_f1_score = f1_score(y_test, pred, average='macro')
    fold_validation_f1 = fold_validation_f1 + bayes_f1_score
    
    
    from sklearn.metrics import precision_score
    bayes_precision_score = precision_score(y_test, pred, average='macro')
    fold_validation_precision = fold_validation_precision + bayes_precision_score
    
    from sklearn.metrics import recall_score
    bayes_recall_score = recall_score(y_test, pred, average='macro')
    fold_validation_recall = fold_validation_recall + bayes_recall_score
    
fold_validation_sum = fold_validation_sum / 10
fold_validation_f1 = fold_validation_f1 / 10
fold_validation_precision = fold_validation_precision / 10
fold_validation_recall = fold_validation_recall / 10

print("Bernoulli Naive Bayes Accuracy Score:",fold_validation_sum)
print("Bernoulli Naive Bayes F1 Score:",fold_validation_f1)
print("Bernoulli Naive Precision Score:",fold_validation_precision)
print("Bernoulli Naive Bayes Recall Score:",fold_validation_recall)
    



