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
    target = data.filter(like="target", axis=1).to_numpy().ravel()

    return features, target

