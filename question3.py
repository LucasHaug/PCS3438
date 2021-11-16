#!/usr/bin/env python3

"""Lasso Regression

Train a Lasso regression, with alpha = 1 and using the leave
one out method for cross validation. The data used to train
and test the regression is the data from the reg01.csv file.

File
-------
question3.py

Author
-------
    Lucas Haug <lucas.haug@usp.br>
"""

from data import get_data
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_predict


def main():
    x_data, y_data = get_data("data/reg01.csv")

    clf = Lasso(alpha=1.0)

    predict = cross_val_predict(clf, x_data, y_data, cv=LeaveOneOut(), n_jobs=4)

    print(f"RMSE: {mean_squared_error(y_data, predict, squared=False)}")


if __name__ == "__main__":
    main()
