#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'PCA Dimension Reduction'
__author__ = 'Fan Shuai'
__mtime__ = '2020/6/28'
"""


import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import math


"""
PCA(copy=True, iterated_power='auto', n_components=2, random_state=None,
  svd_solver='auto', tol=0.0, whiten=False)
"""


def get_data_dr(train_data_address):
    train_data = pd.read_csv(train_data_address)
    X_train = train_data.iloc[:, 0:(train_data.shape[1] - 1)]
    y_train = train_data.iloc[:, (train_data.shape[1] - 1)]

    return X_train, y_train


def dimension_reduction(X_train, n_components):
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)

    pca = PCA(n_components=n_components)
    pca.fit(X_train)
    X_train_pca = pca.transform(X_train)

    return X_train_pca


# results after dimension reduction
def reuslt_handle(X_train_pca, y_train, reduction_result_address):
    X_train_pca = pd.DataFrame(X_train_pca)
    y_train = pd.DataFrame(y_train)
    y_train.columns = ['Species']
    reduction_data = pd.concat([X_train_pca, y_train], axis=1)
    reduction_data.to_csv(reduction_result_address, index=0)


if __name__ == '__main__':
    # a = []
    # for i in range(1, len(sys.argv)):
    #     # a.append((int(sys.argv[i]))), caution 'int' or 'str'
    #     a.append((str(sys.argv[i])))

    a = ["G:\Coding Program\General Algorithm\iris_train.csv",
         "G:\Coding Program\General Algorithm\PCA_results.csv"]

    X_train, y_train = get_data_dr(a[0])
    # amount of dimension after reduction, all/2
    n_components = math.ceil(X_train.shape[1]/2)

    # reduction dimension
    X_train_pca = dimension_reduction(X_train, n_components)
    # output results to csv
    reuslt_handle(X_train_pca, y_train, a[1])
