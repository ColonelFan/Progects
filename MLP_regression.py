#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'MLP Regression'
__author__ = 'Fan Shuai'
__mtime__ = '2020/6/30'
"""


from LinearRegression import get_data, reuslt_handle
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


"""
MLPRegressor(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(100,), learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=None,
       shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
       verbose=False, warm_start=False)
"""


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
    mlp = MLPRegressor()
    mlp.fit(X_train, y_train)
    y_predict = mlp.predict(X_predict)
    y_predict = ss_y.inverse_transform(y_predict)
    # print(y_predict)
    return y_predict


if __name__ == '__main__':
    try:
        # a = []
        # for i in range(1, len(sys.argv)):
        #     # a.append((int(sys.argv[i]))), caution 'int' or 'str'
        #     a.append((str(sys.argv[i])))

        a = [r"G:\Coding Program\General Algorithm\boston_train_regression.csv",
             r"G:\Coding Program\General Algorithm\boston_test_regression.csv",
             r"G:\Coding Program\General Algorithm\MLP_results.csv"]

        X_train, y_train, X_predict = get_data(a[0], a[1])

        # predict
        y_predict = prdedict_mode(X_train, y_train, X_predict)
        # output results to csv
        reuslt_handle(X_predict, y_predict, a[2])
        print_result = "1"
    except:
        print_result = "0"

    print(print_result)
