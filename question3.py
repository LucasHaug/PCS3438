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
import numpy as np


def main():
    x_data, y_data = get_data("data/reg01.csv")

    cv = LeaveOneOut()

    rmse_train = []
    rmse_test = []

    for train_index, test_index in cv.split(x_data):
        # Split data
        x_train, x_test = x_data[train_index, :], x_data[test_index, :]
        y_train, y_test = y_data[train_index], y_data[test_index]

        # Fit model
        model = Lasso(alpha=1.0)
        model.fit(x_train, y_train)

        # Evaluate model
        train_predict = model.predict(x_train)
        test_predict = model.predict(x_test)

        # Calculate rmse
        rmse_train.append(mean_squared_error(y_train, train_predict, squared=False))
        rmse_test.append(mean_squared_error(y_test, test_predict, squared=False))

    print("Root Mean Squared Errors:")
    print(f"Mean Train RMSE: {np.mean(rmse_train) : .2f}")
    print(f"Mean Test RMSE:  {np.mean(rmse_test) : .2f}")


if __name__ == "__main__":
    main()
