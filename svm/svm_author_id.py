#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
from tools.email_preprocess import preprocess
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import numpy as np
sys.path.append("../tools/")

# features_train and features_test are the features for the training
# and testing datasets, respectively
# labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()
clf = SVC(kernel="rbf", C=10000.)
start = time()
# features_train = features_train[:len(features_train) // 100]
# labels_train = labels_train[:len(labels_train) // 100]
clf.fit(features_train, labels_train)
print("Time taken for training the model: ", round(time() - start, 3), "s\n")

start = time()
pred = clf.predict(features_test)
print("Time taken for predicting: ", round(time() - start, 3), "s\n")

# print("accuracy: ", accuracy_score(pred, labels_test))
print(np.count_nonzero(pred))




