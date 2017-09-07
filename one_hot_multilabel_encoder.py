#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from collections import Counter, OrderedDict

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import matplotlib.pyplot as plt


class CategoricalFeaturesEncoder(object):
    """
    Wraps sklearn LabelEncoder functionality for use on multiple columns of a
    pandas dataframe.

    """
    def __init__(self, categorical_features=None):
        self.columns = np.array(categorical_features, dtype=object)

    def fit(self, dframe):
        """
        Fit label encoder to pandas columns.

        Access individual column classes via indexig `self.all_classes_`

        Access individual column encoders via indexing
        `self.all_encoders_`
        """
        self.fit_transform(dframe)        
        return self

    def fit_transform(self, dframe):
        """
        Fit label encoder and return encoded labels.

        Access individual column classes via indexing
        `self.all_classes_`

        Access individual column encoders via indexing
        `self.all_encoders_`

        Access individual column encoded labels via indexing
        `self.all_labels_`
        """
        dframe = dframe.copy()
        self.dframe_fit = dframe.copy()
        # if columns are provided, iterate through and get `classes_`
        if self.columns is None:
            # no columns specified; assume all are to be encoded
            self.columns = dframe.iloc[:, :].columns
        
        # ndarray to hold LabelEncoder().classes_ for each
        # column; should match the shape of specified `columns`
        self.all_classes_ = np.ndarray(shape=self.columns.shape,
                                       dtype=object)
        self.all_encoders_ = np.ndarray(shape=self.columns.shape,
                                        dtype=object)
        self.all_labels_ = np.ndarray(shape=self.columns.shape,
                                      dtype=object)
        self.all_frequencies_ = np.ndarray(shape=self.columns.shape,
                                      dtype=object)
        self.index_to_column_ = OrderedDict()
        
        for idx, column in enumerate(self.columns):
            col_values = dframe.loc[:, column].values

            self.all_frequencies_[idx] = (column,
                                          Counter(col_values))
            self.index_to_column_[idx] = column   
            # instantiate LabelEncoder
            le = LabelEncoder()
            # fit and transform labels in the column
            dframe.loc[:, column] =\
                le.fit_transform(col_values)
            # append the `classes_` to our ndarray container
            self.all_classes_[idx] = (column,
                                      np.array(le.classes_.tolist(),
                                              dtype=object))
            self.all_encoders_[idx] = le
            self.all_labels_[idx] = le
        
        return dframe

        
    def get_index(self, column):
        assert column in self.columns
        for idx, col in self.index_to_column_.items():
            if col == column:
                return idx
        
    def get_column_name(self, idx):
        return self.index_to_column_[idx]        
                
    def transform(self, dframe):
        """
        Transform labels to normalized encoding.
        """
        dframe = dframe.copy()
        
        if self.columns is None:
            self.columns = np.array(dframe.columns, dtype=object)
            
        for idx, column in enumerate(self.columns):
            dframe.loc[:, column] = \
            self.all_encoders_[idx].transform(dframe.loc[:, column].values)

        return dframe  #.loc[:, self.columns].values

    def inverse_transform(self, dframe):
        """
        Transform labels back to original encoding.
        """
        dframe = dframe.copy()
        if self.columns is not None:
            self.columns = np.array(dframe.columns, dtype=object)
        
        for idx, column in enumerate(self.columns):
            dframe.loc[:, column] = self.all_encoders_[idx]\
                            .inverse_transform(dframe.loc[:, column].values)
        return dframe


    def plot_histogram(self, column_name, bins=None, color='b', normed=False):
        assert column_name in self.columns

        idx = self.get_index(column_name)
        col, counter = self.all_frequencies_[idx]
        assert col == column_name

        labels, frequencies = list(counter.keys()), list(counter.values())
        
        le = self.all_encoders_[idx]
        mask_integers = le.transform(self.all_classes_[idx][1])

#        if bins is None:
#            bins = len(self.dframe_fit[column_name].value_counts())
        
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.bar(mask_integers, frequencies, color=color) 
#        ax.hist(self.dframe_fit[column_name],
#                                   bins = bins,
#                                   normed = normed)
        ax.set_xlabel(column_name)
        ylabel = 'Frequency' + (' number' if normed==False else '')
        ax.set_ylabel(ylabel)

        ax.set_xticks(mask_integers)

        labels = list(map(lambda s: s[:20], labels))
        ax.set_xticklabels(labels, rotation=20)

        
        

class OneHotEncoder_(CategoricalFeaturesEncoder):
    """
    Wraps sklearn OneHotEncoder functionality for use on multiple columns of a
    pandas dataframe. First passes through MultiColumnLabelEncoder

    Parameters
    ----------
    n_values : 'auto', int or array of ints
        Number of values per feature.

        - 'auto' : determine value range from training data.
        - int : number of categorical values per feature.
                Each feature value should be in ``range(n_values)``
        - array : ``n_values[i]`` is the number of categorical values in
                  ``X[:, i]``. Each feature value should be
                  in ``range(n_values[i])``

    categorical_features : "all" or array of indices or mask
        Specify what features are treated as categorical.

        - 'all' (default): All features are treated as categorical.
        - array of indices: Array of categorical feature indices.
        - mask: Array of length n_features and with dtype=bool.

        Non-categorical features are always stacked to the right of the matrix.

    sparse : boolean, default=False
        Will return sparse matrix if set True else will return an array.
    
    """
    
    def __init__(self, columns=None, sparse=False,
                                     n_values='auto'):
        super(self.__class__, self).__init__(columns)
        self.one_hot_encoder = OneHotEncoder(sparse=sparse,
                                             n_values=n_values,
                                             categorical_features=columns)
        
        
    def fit(self, dframe):        
        categorical_inds = [idx for idx, name in enumerate(dframe) \
                                if name in self.columns]
        self.one_hot_encoder.categorical_features = categorical_inds
        self.one_hot_encoder.fit(super(self.__class__, self) \
                            .fit_transform(dframe))
        
    def fit_transform(self, dframe):
        categorical_inds = [idx for idx, name in enumerate(dframe) \
                                if name in self.columns]
        self.one_hot_encoder.categorical_features = categorical_inds
        res = self.one_hot_encoder\
                  .fit_transform(super(self.__class__, self) \
                  .fit_transform(dframe))
        return res
    
    def transform(self, dframe):
        return self.one_hot_encoder.transform(super(self.__class__, self) \
                                    .transform(dframe))
        
if __name__ == "__main__":
    
    # Small unit test
    fruit_data = pd.DataFrame({
    'fruit':  ['apple','orange','pear','lemon'],
    'color':  ['red','orange','green','green'],
    'weight': [5,6,3,4],
    'price': [1.3, 2.0, 2.5, 3.2]
    })

    fruit_data1 = pd.DataFrame({
    'fruit':  ['pear','lemon', 'pear', 'lemon'],
    'color':  ['green','green', 'red', 'green'],
    'weight': [3,4, 2,3],
    'price': [2.5, 3.2, 2.4, 5.5]
    })

    
    columns = fruit_data.select_dtypes(include=[object]).columns
    encoder = CategoricalFeaturesEncoder(columns)    
    
    
    
    res = encoder.fit_transform(fruit_data)    
    res1 = encoder.transform(fruit_data)

#    s = plt.hist(fruit_data['fruit'], bins=4, normed=True)

    
    encoder.plot_histogram(column_name='color')
    
    
    one_hot_enc = OneHotEncoder_(columns)
    res2 = one_hot_enc.fit_transform(fruit_data)
    res3 = one_hot_enc.transform(fruit_data)
    
    
    
    train_data = fruit_data
    test_data = fruit_data
    X_train = pd.get_dummies(train_data)
    X_test = pd.get_dummies(test_data)
    X_test.reindex(columns = X_train.columns, fill_value = 0)
    
    
    