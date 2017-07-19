#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from tools.email_preprocess import preprocess
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

# features_train and features_test are the features for the training
# and testing datasets, respectively
# labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()
# fit
clf = GaussianNB()
start = time()
clf.fit(features_train, labels_train)
print("training time: ", round(time() - start, 3), "s")
# predict
start = time()
pred = clf.predict(features_test)
print("predicting time: ", round(time() - start, 3), "s")

# print(accuracy_score(pred, labels_test))





