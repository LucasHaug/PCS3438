#!/usr/bin/env python3

"""Gaussian Naive Bayes classifier

Train a Gaussian Naive Bayes classifier using the holdout
method for cross validation. The data used to train and
test the classifier is the data from the class01.csv file.

File
-------
question1.py

Author
-------
    Lucas Haug <lucas.haug@usp.br>
"""

from data import get_data
from sklearn.naive_bayes import GaussianNB


def get_holdout_data(file_name, holdout_size):
    """
    Reads the data from the file and returns the training
    and test sets using the holdout method.

    Args:
        file_name (str): The name of the file to be read.
        holdout_size (int): The size of the holdout set.
    """

    features, target = get_data(file_name)

    x_train = features[:-holdout_size]
    y_train = target[:-holdout_size].ravel()
    x_test = features[-holdout_size:]
    y_test = target[-holdout_size:].ravel()

    return x_train, y_train, x_test, y_test


def main():
    x_train, y_train, x_test, y_test = get_holdout_data(
        "data/class01.csv", holdout_size=650
    )

    clf = GaussianNB()
    clf.fit(x_train, y_train)

    print(f"Train Set Accuracy: {clf.score(x_train, y_train) * 100 : .2f}%")
    print(f"Test Set Accuracy:  {clf.score(x_test, y_test) * 100 : .2f}%")


if __name__ == "__main__":
    main()
