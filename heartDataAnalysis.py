# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 16:00:11 2020

@author: endi_
"""

#imports
import os
import pandas as pd
import numpy as np

from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

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

#columns to plot on the confusion matrix
df.columns = ['age', 'sex',	'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope',	'ca', 'thal', 'target']
cols = ['age', 'sex']

#Encode the feature values which are strings to integers
for label in df.columns:
    df[label] = LabelEncoder().fit(df[label]).transform(df[label])

# Create our X and y data    
result = []
for x in df.columns:
    if x != 'target':
        result.append(x)

X = df[result].values
y = df['target'].values

#split the dataset
X_train, X_test, y_train, y_test = train_test_split(    
    X, y, test_size=0.2)

#build a knn model
knn = KNeighborsClassifier(n_neighbors=52, weights='uniform')
knn.fit(X_train, y_train) 

#Instantiate the model with 10 trees and entropy as splitting criteria
rf = RandomForestClassifier(n_estimators = 40)
rf.fit(X_train, y_train)

#make predictions
y_pred_knn = knn.predict(X_test)
y_pred_rf = rf.predict(X_test)

#print results
print('kNN Accuracy: %.3f' % accuracy_score(y_test, y_pred_knn))
print('Random Forest Accuracy: %.3f' % accuracy_score(y_test, y_pred_rf))

#print confusion matrix numerically, using library method
cm = confusion_matrix(y_test, y_pred_knn)
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

#build a new data frame with three columns, the actual values of the test data, 
#and the predictions of the model
df_compare = pd.DataFrame({'Actual': y_test, 'Predicted (kNN)': y_pred_knn, 'Predicted (RF)': y_pred_rf})