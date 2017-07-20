#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
from tools.email_preprocess import preprocess
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

sys.path.append("../tools/")

# features_train and features_test are the features for the training
# and testing datasets, respectively
# labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

# your code goes here
print(features_train.shape)
clf = DecisionTreeClassifier(min_samples_split=40)

# features_train = features_train[:len(features_train) // 100]
# labels_train = labels_train[:len(labels_train) // 100]


start = time()
clf.fit(features_train, labels_train)
print("Time taken for training the model: ", round(time() - start, 3), "s\n")

start = time()
pred = clf.predict(features_test)
print("Time taken for predicting the result: ", round(time() - start, 3), "s\n")

print(accuracy_score(labels_test, pred))





