#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'SVD Dimension Reduction'
__author__ = 'Fan Shuai'
__mtime__ = '2020/6/30'
"""


import numpy as np
import math
from numpy import linalg
from PCA_dimension_reduction import get_data_dr, reuslt_handle


def dimension_reduction(X_train, n_components):
    A = np.mat(X_train)
    U, Sigma, VT = linalg.svd(A)
    Sigma[n_components:len(Sigma)] = 0  #dimension reduction setting

    S = np.zeros((len(A), len(VT)))
    S[:len(VT), :len(VT)] = np.diag(Sigma)
    # X_train_conv = np.dot(np.dot(A.T, U), S)# dimension reduction
    X_train_conv = np.dot(np.dot(U, S), VT)  # dimension reduction

    return X_train_conv


if __name__ == '__main__':
    try:
        # a = []
        # for i in range(1, len(sys.argv)):
        #     # a.append((int(sys.argv[i]))), caution 'int' or 'str'
        #     a.append((str(sys.argv[i])))

        a = ["G:\Coding Program\General Algorithm\iris_reduction_dimension.csv",
             "G:\Coding Program\General Algorithm\SVD_results.csv"]

        X_train = get_data_dr(a[0])
        # amount of dimension after reduction, all/2
        n_components = math.ceil(X_train.shape[1]/2)

        # reduction dimension
        X_train_conv = dimension_reduction(X_train, n_components)
        # output results to csv
        reuslt_handle(X_train_conv, a[1])
        print_result = "1"
    except:
        print_result = "0"

    print(print_result)
