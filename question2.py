#!/usr/bin/env python3

"""KNN classifier

Train a KNN classifier, which uses k = 10 and euclidean
distance, using k-fold, with 5 folds, for cross validation.
The data used to train and test the classifier is the data
from the class02.csv file.

File
-------
question2.py

Author
-------
    Lucas Haug <lucas.haug@usp.br>
"""

from data import get_data
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


def main():
    x_data, y_data = get_data("data/class02.csv")

    clf = KNeighborsClassifier(n_neighbors=10, metric="euclidean", n_jobs=4)
    clf.fit(x_data, y_data)

    scores = cross_validate(clf, x_data, y_data, cv=5, n_jobs=4)

    print(f"Mean accuracy: {np.mean(scores['test_score']) * 100 : .2f}%")


if __name__ == "__main__":
    main()
