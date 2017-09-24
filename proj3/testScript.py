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
fig, sub = plt.subplots(3, 3)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

X0, X1 = X_test[:, 0], X_test[:, 1]
xx, yy = make_meshgrid(X0, X1)

for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y_test, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)



#clf = models[0].fit(X_train, y_train)
(clf.score(X_test, y_test) for clf in models)
                           















plt.show()



