# By Daniel Michelin
# Assistance from https://youtu.be/tNa99PG8hR8

# Imports the iris dataset and makes predictions

from sklearn.datasets import load_iris
import numpy as np
from sklearn import tree

# From the first part of the video

iris = load_iris()
print (iris.feature_names) # Prits features of the data set
print (iris.target_names) # Prints possible outputs 
print (iris.data[0]) # Prints the first entry in the dataset
print ("")

# From the second part of the video

test_idx = [0,50,100] # We have an ordered dataset, the first setosa at index 0,
                        # the first versicolor at 50, and virginica at 100

# training data
train_target = np.delete(iris.target, test_idx) #From the dataset, remove three
                                                # entries specified by test_idx
                                                # to train target outputs

train_data = np.delete(iris.data, test_idx, axis=0) #Do the same thing as above,  
                                                    # but find target data

# testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier() # Make the classifier a decision tree
clf.fit(train_data, train_target) # Train the classifier

print (test_target) # print the test targets
print (clf.predict(test_data)) # predict the targets using the test data

# I couldn't get the graph to work because pydot couldn't be imported for python
# 3 on my computer