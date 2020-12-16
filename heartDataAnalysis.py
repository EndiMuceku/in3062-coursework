# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 16:00:11 2020

@author: endi_
"""

#imports
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt

#plot a confusion matrix
def plot_confusion_matrix(cm, names, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar(fraction=0.05)
    tick_marks = np.arange(len(names))
    plt.xticks(tick_marks, names, rotation=45)
    plt.yticks(tick_marks, names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#load the dataset
path = "."

filename_read = os.path.join(path, "heart.csv")
df = pd.read_csv(filename_read)
df - df.reindex(np.random.permutation(df.index))

#get X and y values
X = df.drop('target', axis=1)
y = df.target.values

#columns to plot on the confusion matrix
cols = ['age', 'sex']

#split the dataset
X_train, X_test, y_train, y_test = train_test_split(    
    X, y, test_size=0.2, random_state=9) 

#build a knn model
knn = KNeighborsClassifier()
knn.fit(X_train, y_train) 

y_pred = knn.predict(X_test)

#print results
print('kNN Accuracy: %.3f' % accuracy_score(y_test, y_pred))

#print confusion matrix numerically, using library method
cm = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)
print('Confusion matrix, without normalization')
print(cm)
plt.figure()
plot_confusion_matrix(cm, cols, title='')

#graphical plots of confusion matrix using method above
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)
plt.figure()
plot_confusion_matrix(cm_normalized, cols, title='Normalized confusion matrix')
plt.show()