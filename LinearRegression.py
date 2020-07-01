#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'Linear Regression with Least Square'
__author__ = 'Fan Shuai'
__mtime__ = '2020/6/30'
"""


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


"""
 LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
"""


def get_data(train_data_address, predict_data_address):
    train_data = pd.read_csv(train_data_address)
    X_train = train_data.iloc[:, 0:(train_data.shape[1] - 1)]
    y_train = train_data.iloc[:, (train_data.shape[1] - 1)]

    predict_data = pd.read_csv(predict_data_address)
    X_predict = predict_data.iloc[:, 0:predict_data.shape[1]]
    return X_train, y_train, X_predict


def prdedict_mode(X_train, y_train, X_predict):
    # normalize
    ss_X = StandardScaler()
    X_train = ss_X.fit_transform(X_train)
    X_predict = ss_X.transform(X_predict)

    ss_y = StandardScaler()
    y_train = np.array(y_train).reshape(-1, 1)
    y_train = ss_y.fit_transform(y_train)
    y_train = np.ravel(y_train)

    # train and predict
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_predict = lr.predict(X_predict)
    y_predict = ss_y.inverse_transform(y_predict)
    # print(y_predict)
    return y_predict


# results has independent variables and dependent variables
def reuslt_handle(X_predict, y_predict, predict_result_address):
    y_predict = pd.DataFrame(y_predict)
    y_predict.columns = ['result']
    predict_data = pd.concat([X_predict, y_predict], axis=1)
    predict_data.to_csv(predict_result_address, index=0)


if __name__ == '__main__':
    try:
        # a = []
        # for i in range(1, len(sys.argv)):
        #     # a.append((int(sys.argv[i]))), caution 'int' or 'str'
        #     a.append((str(sys.argv[i])))

        a = [r"G:\Coding Program\General Algorithm\boston_train_regression.csv",
             r"G:\Coding Program\General Algorithm\boston_test_regression.csv",
             r"G:\Coding Program\General Algorithm\LR_results.csv"]

        X_train, y_train, X_predict = get_data(a[0], a[1])

        # predict
        y_predict = prdedict_mode(X_train, y_train, X_predict)
        # output results to csv
        reuslt_handle(X_predict, y_predict, a[2])
        print_result = "1"
    except:
        print_result = "0"

    print(print_result)
