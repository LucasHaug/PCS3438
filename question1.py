#!/usr/bin/env python3

"""Train a Gaussian Naive Bayes classifier using the data from class01.csv.
File
-------
question1.py

Author
-------
    Lucas Haug <lucas.haug@usp.br>
"""

from data import get_holdout_data
from sklearn.naive_bayes import GaussianNB


def main():
    x_train, y_train, x_test, y_test = get_holdout_data("data/class01.csv", holdout_size=650)

    clf = GaussianNB()
    clf.fit(x_train, y_train)

    print(f"Train Set Accuracy: {clf.score(x_train, y_train) * 100 : .2f}%")
    print(f"Test Set Accuracy:  {clf.score(x_test, y_test) * 100 : .2f}%")


if __name__ == "__main__":
    main()