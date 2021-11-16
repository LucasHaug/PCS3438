#!/usr/bin/env python3

"""Decision Tree Regression

Train a Decision Tree regression, with no leaves pruning, and using
using k-fold, with 5 folds, for cross validation. The data used to
train and test the regression is the data from the reg02.csv file.

File
-------
question4.py

Author
-------
    Lucas Haug <lucas.haug@usp.br>
"""

from data import get_data
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
import numpy as np


def main():
    x_data, y_data = get_data("data/reg02.csv")

    cv = KFold(n_splits=5)

    mae_train = []
    mae_test = []

    for train_index, test_index in cv.split(x_data):
        # Split data
        x_train, x_test = x_data[train_index, :], x_data[test_index, :]
        y_train, y_test = y_data[train_index], y_data[test_index]

        # Fit model
        model = DecisionTreeRegressor(criterion="absolute_error")
        model.fit(x_train, y_train)

        # Evaluate model
        train_predict = model.predict(x_train)
        test_predict = model.predict(x_test)

        # Calculate rmse
        mae_train.append(mean_absolute_error(y_train, train_predict))
        mae_test.append(mean_absolute_error(y_test, test_predict))

    print("Mean Absolute Errors:")
    print(f"Mean Train MAE: {np.mean(mae_train)}")
    print(f"Mean Test MAE:  {np.mean(mae_test)}")


if __name__ == "__main__":
    main()
