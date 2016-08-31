# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 11:56:34 2016

@author: Daniel

with assistance from https://youtu.be/84gqSbLcBFE
"""

from sklearn import datasets
iris = datasets.load_iris()

X = iris.data
y = iris.target

from sklearn.cross_validation import train_test_split

#Split the dataset in half for y and x, one half for training and the other for
#test
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = .5) 

# from sklearn import tree
# my_classifier = DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier
my_classifier = KNeighborsClassifier()

my_classifier.fit(X_train, y_train)

predictions = my_classifier.predict(X_test)
# print (predictions)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,predictions))

