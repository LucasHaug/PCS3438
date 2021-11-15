#!/usr/bin/env python3

import pandas as pd


def get_data(file_name):
    """
    Reads the data from the file and returns the features and the target.

    Args:
        file_name (str): The name of the file to be read.
    """

    data = pd.read_csv(file_name)
    features = data.filter(like="x", axis=1).to_numpy()
    target = data.filter(like="target", axis=1).to_numpy()

    return features, target


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
