#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
Filename: data
Date: 2019-10-06 12:28
Project: mymldev
AUTHOR: Suneel Dondapati
"""

from dataclasses import dataclass, field
import pandas as pd


SEED = 1310


@dataclass
class XYData:
    """Splits input data to IDVs and DV.

    Parameters
    ----------
    data: pandas.DataFrame
        Input pandas dataframe.
    x_cols: list
        IDV column names.
    y_col: str
        DV column name.
    type_of_data: str
        'Train' or 'Validation' or 'Test'

    Attributes
    ----------
    data: pandas.DataFrame
        Input pandas dataframe.
    x_cols: list
        IDV column names.
    y_col: str
        DV column name.
    type_of_data: str
        'Train' or 'Validation' or 'Test'
    X: pandas.DataFrame
        IDVs.
    y: pandas.Series
        DV.
    excluded_X: pandas.DataFrame
        Columns excluded from X
    """
    data: pd.DataFrame = field(repr=False)
    x_cols: list = field(repr=False)
    y_col: str
    type_of_data: str = field(default='Train')
    X: pd.DataFrame = field(init=False, repr=False)
    y: pd.Series = field(init=False, repr=False)
    excluded_X: pd.DataFrame = field(init=False, repr=False)

    def __post_init__(self):
        if len(set(self.data.columns)) != len(self.data.columns):
            raise ValueError("Duplicate columns are not allowed in the data")
        self.X = self.data[self.x_cols].copy()
        self.y = self.data[self.y_col].copy()
        self.excluded_X = self.data[list(set(self.data.columns) - set(self.x_cols) - set([self.y_col]))]


def get_binary_classifier_train_test_data(data: pd.DataFrame, y_col: str,
                                          testset_size: float=0.25,
                                          per_0_1: list = [90, 10],
                                          excess_data: bool = False):
    """Split data into train and test

    Parameters
    ----------
    data: pandas.DataFrameobject
        Model data.
    y_col: str
        DV column name.
    testset_size: float
        Test data size
    per_0_1:
        List[% of 0's, % of 1's] for training data.
        Data-sampling is performed assuming 1's are less
        compared to 0's.
    excess_data : bool, default is False
        Returns excess data after sampling if True.

    Returns
    -------
    train: pandas.DataFrame
        Training data
    test: pandas.DataFrame
        Test data
    leftover_data: pandas.DataFrame
    """
    try:
        from sklearn.cross_validation import train_test_split
    except ModuleNotFoundError:
        from sklearn.model_selection import train_test_split
    train, test = train_test_split(data, test_size=testset_size,
                                   stratify=data[y_col], random_state=SEED)
    zeros = train[train[y_col] == 0].copy()
    ones = train[train[y_col] == 1].copy()
    sampled_zeros = zeros.sample(int((len(ones) * per_0_1[0]) / per_0_1[1]),
                                 replace=False, random_state=SEED)
    leftover_data= zeros.iloc[zeros.index[~zeros.index.isin(sampled_zeros.index)]]
    train = pd.concat([sampled_zeros, ones], axis=0, ignore_index=True)
    if excess_data:
        return train, test, leftover_data
    return train, test

























