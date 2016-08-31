# By Daniel Michelin
# Assistance from https://youtu.be/cKxRvEZd3Mw

from sklearn import tree
features = [[140,1], [130,1], [140,0], [170,0]] # Syntax is [weight(in grams),type of fruit (0 = orange, 1 = apple)]
labels = [0,0,1,1]
clf = tree.DecisionTreeClassifier() # classifier
clf = clf.fit(features,labels) # training the classifier
print (clf.predict([[150,0]]))