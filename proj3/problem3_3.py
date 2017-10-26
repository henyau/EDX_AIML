"""
This script demonstrates using sklearn using support vector machine (SVM) 
classifier using different kernels and other classifiers

Usage: python problem3_3.py input3.csv 

where input3.csv has three columns, real data A and B and a label of 0 or 1

"""

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

import csv
import pandas as pd
import sys

def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

#
#with open('input3.csv', newline='') as csvfile:
#    input3 = csv.reader(csvfile, delimiter=' ', quotechar='|')
#    for row in input3:
#        print(', '.join(row))
#      


input3 = pd.read_csv(sys.argv[1])
input3.pop('label')
X = input3.values

input3 = pd.read_csv(sys.argv[1])
input3.pop('A')
input3.pop('B')
y = (input3.values)
y = np.ravel(y)


#X = []
#y = []
#with open(sys.argv[1], newline='') as csvfile:
#     inputcsv = csv.reader(csvfile, delimiter=',', quotechar='|')
#     for row in inputcsv:
#         rowInt = []
#         for elem in row:
#             float1 = float(elem)
#             rowInt.append(float1)
#         X.append(rowInt[0:2])
#         y.append(([rowInt[2]]))
#         
#X = np.array(X)
#y = np.array(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42, stratify=y)
#divide into Xtrain and Xtest. 60/40




# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 1.0  # SVM regularization parameter
models = (svm.SVC(kernel='linear', C=0.1),
          svm.SVC(kernel='poly', degree=5, C=1,gamma = 0.5),
          svm.SVC(kernel='rbf', gamma=0.7, C=5),
          LogisticRegression(C=5),
          KNeighborsClassifier(n_neighbors = 5, leaf_size = 15),
          DecisionTreeClassifier(max_depth=5,min_samples_split=4),
          RandomForestClassifier(max_depth=5, min_samples_split=4 ))
models = (clf.fit(X_train, y_train) for clf in models)

# title for the plots
titles = ('SVC with linear kernel',
          'SVC with polynomial kernel',
          'SVC with RBF kernel',
          'Logistic Regression',
          'KNN',
          'DecisionTree',
          'RandomForest')

# Set-up 2x2 grid for plotting.
fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)

for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

plt.show()