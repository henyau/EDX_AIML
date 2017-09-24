print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

import csv
import pandas as pd

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


input3 = pd.read_csv("input3.csv")
input3.pop('label')
X = input3.values

input3 = pd.read_csv("input3.csv")
input3.pop('A')
input3.pop('B')
y = (input3.values)
y = np.ravel(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42, stratify=y)
#divide into Xtrain and Xtest. 60/40



##C = 5.0  # SVM regularization parameter
models = (svm.SVC(kernel='linear', C=0.1),
          svm.SVC(kernel='poly', degree=5, C=1,gamma = 0.5),
          svm.SVC(kernel='rbf', gamma=0.5, C=100),
          LogisticRegression(C=0.1),
          KNeighborsClassifier(n_neighbors = 1, leaf_size = 5),
          DecisionTreeClassifier(max_depth=6,min_samples_split=2),
          RandomForestClassifier(max_depth=22, min_samples_split=5 ))
##models = (clf.fit(X_train, y_train) for clf in models)

clf = models[0].fit(X_train, y_train)
print(clf.score(X_train, y_train))

clf = models[1].fit(X_train, y_train)
print(clf.score(X_train, y_train))

clf = models[2].fit(X_train, y_train)
print(clf.score(X_train, y_train))

clf = models[3].fit(X_train, y_train)
print(clf.score(X_train, y_train))

clf = models[4].fit(X_train, y_train)
print(clf.score(X_train, y_train))

clf = models[5].fit(X_train, y_train)
print(clf.score(X_train, y_train))

clf = models[6].fit(X_train, y_train)
print(clf.score(X_train, y_train))

clf = models[0].fit(X_train, y_train)
print(clf.score(X_test, y_test))

clf = models[1].fit(X_train, y_train)
print(clf.score(X_test, y_test))

clf = models[2].fit(X_train, y_train)
print(clf.score(X_test, y_test))

clf = models[3].fit(X_train, y_train)
print(clf.score(X_test, y_test))

clf = models[4].fit(X_train, y_train)
print(clf.score(X_test, y_test))

clf = models[5].fit(X_train, y_train)
print(clf.score(X_test, y_test))

clf = models[6].fit(X_train, y_train)
print(clf.score(X_test, y_test))