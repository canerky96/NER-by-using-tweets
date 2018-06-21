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

from sklearn.ensemble import RandomForestClassifier
rf_classifier=RandomForestClassifier(n_estimators=10)

t0=time()
rf_classifier=rf_classifier.fit(variables,labels)

training_time_container['random_forest']=time()-t0
print("Training Time: %fs"%training_time_container['random_forest'])

t0=time()
rf_predictions=rf_classifier.predict(variables)
prediction_time_container['random_forest']=time()-t0
print("Prediction Time: %fs"%prediction_time_container['random_forest'])

accuracy_container['random_forest']=sklearn.metrics.accuracy_score(labels, rf_predictions)
print ("Accuracy Score of Random Forests Classifier: ")
print(accuracy_container['random_forest'])


from sklearn.metrics import f1_score
random_forest_f1_score = f1_score(labels, rf_predictions, average='macro')
print("Random Forest F1 Score:",random_forest_f1_score)

from sklearn.metrics import precision_score
random_forest_precision_score = precision_score(labels, rf_predictions, average='macro')
print("Random Forest Precision Score:",random_forest_precision_score)

from sklearn.metrics import recall_score
random_forest_recall_score = recall_score(labels, rf_predictions, average='macro')
print("Random Forest Recall Score:",random_forest_recall_score)


################################################################################

variables_train, variables_test, labels_train, labels_test=train_test_split(
        variables, labels, test_size=.9)

from sklearn.ensemble import RandomForestClassifier
rf_classifier=RandomForestClassifier(n_estimators=10)
rf_classifier=rf_classifier.fit(variables_train,labels_train)
rf_predictions=rf_classifier.predict(variables_test)


random_forest_accuracy_score=sklearn.metrics.accuracy_score(labels_test, rf_predictions)
print("Random Forest Accuracy Score:",random_forest_accuracy_score)

from sklearn.metrics import f1_score
random_forest_f1_score = f1_score(labels_test, rf_predictions, average='macro')
print("Random Forest F1 Score:",random_forest_f1_score)

from sklearn.metrics import precision_score
random_forest_precision_score = precision_score(labels_test, rf_predictions, average='macro')
print("Random Forest Precision Score:",random_forest_precision_score)

from sklearn.metrics import recall_score
random_forest_recall_score = recall_score(labels_test, rf_predictions, average='macro')
print("Random Forest Recall Score:",random_forest_recall_score)

################################################################################

fold_validation_sum = 0
fold_validation_f1 = 0
fold_validation_precision = 0
fold_validation_recall = 0

from sklearn.model_selection import KFold, cross_val_score
k_fold = KFold(n_splits=10,shuffle=True)
for train_indices, test_indices in k_fold.split(variables):
    x_train, x_test = variables[train_indices], variables[test_indices]
    y_train, y_test = labels[train_indices], labels[test_indices]
    
    rf_classifier=RandomForestClassifier(n_estimators=10)
    rf_classifier=rf_classifier.fit(x_train,y_train)
    pred=rf_classifier.predict(x_test)
    

    accuracy=sklearn.metrics.accuracy_score(y_test, pred)
    fold_validation_sum = fold_validation_sum + accuracy
    
    from sklearn.metrics import f1_score
    rf_f1_score = f1_score(y_test, pred, average='macro')
    fold_validation_f1 = fold_validation_f1 + rf_f1_score
    
    
    from sklearn.metrics import precision_score
    rf_precision_score = precision_score(y_test, pred, average='macro')
    fold_validation_precision = fold_validation_precision + rf_precision_score
    
    from sklearn.metrics import recall_score
    rf_recall_score = recall_score(y_test, pred, average='macro')
    fold_validation_recall = fold_validation_recall + rf_recall_score
    
fold_validation_sum = fold_validation_sum / 10
fold_validation_f1 = fold_validation_f1 / 10
fold_validation_precision = fold_validation_precision / 10
fold_validation_recall = fold_validation_recall / 10

print("Random Forest Accuracy Score:",fold_validation_sum)
print("Random Forest F1 Score:",fold_validation_f1)
print("Random Forest Precision Score:",fold_validation_precision)
print("Random Forest Recall Score:",fold_validation_recall)

