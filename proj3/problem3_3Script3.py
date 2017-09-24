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
          svm.SVC(kernel='rbf', gamma=0.7, C=5),
          LogisticRegression(C=5),
          KNeighborsClassifier(n_neighbors = 5, leaf_size = 15),
          DecisionTreeClassifier(max_depth=5,min_samples_split=4),
          RandomForestClassifier(max_depth=5, min_samples_split=4 ))
##models = (clf.fit(X_train, y_train) for clf in models)

# title for the plots

##tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
##                     'C': [1, 10, 100, 1000]},
##                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

v1 = list(range(1,50))

calc = 5
v2 =[]
while int(calc) < 61:
    v2.append(calc)
    calc = int(calc) + 5
v3 = list(range(2,10))


tuned_parameters = [{'C': [0.1, 0.5, 1, 5, 10, 50, 100]}]
##[{'max_depth': v1, 'min_samples_split' : v3}]
#[{'n_neighbors': v1, 'leaf_size' : v2}]
                       #{'n_neighbors':[1, 2, 3, ..., 50], 'leaf_size' : [5, 10, 15, ..., 60]},
                       #{'max_depth': [1, 2, 3, ..., 50], 'min_samples_split': [2, 3, 4, ..., 10]}]
 
scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(LogisticRegression(), tuned_parameters, cv=5,
                       scoring='%s_macro' % score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()


