# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 13:15:29 2016

@author: Daniel

with assistance from: https://youtu.be/AoeEHqVSNOw
"""
from scipy.spatial import distance
#import random

#defines a function to calculate the euclidian distance between and 'a' and 'b'
def euc(a, b):
    return distance.euclidean(a,b)
    
# Creating our own barebones classifier using our two methods
class ScrappyKNN():
    def fit(self, X_train, y_train):
        self.X_train = X_train 
        self.y_train = y_train
    def predict(self,X_test):
        predictions = [] # We must return an array
        for row in X_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions
        
    def closest(self, row):
       best_dist = euc(row, self.X_train[0]) # default value for the closest distance
       best_index = 0 # default value for the index with the closest value
       for i in range(1, len(self.X_train[i])):
          dist = euc (row, self.X_train[i])
          if dist < best_dist:
              best_index = i
              best_dist = dist
              
       return self.y_train[best_index]
        
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

#from sklearn.neighbors import KNeighborsClassifier
my_classifier = ScrappyKNN()

my_classifier.fit(X_train, y_train)

predictions = my_classifier.predict(X_test)
# print (predictions)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,predictions))

