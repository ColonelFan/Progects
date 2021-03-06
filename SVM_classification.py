#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'SVM Classification'
__author__ = 'Fan Shuai'
__mtime__ = '2020/6/30'
"""


import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


"""
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
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
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_predict = ss.transform(X_predict)

    svm = SVC()
    svm.fit(X_train, y_train)
    y_predict = svm.predict(X_predict)
    # print(y_predict)
    return y_predict


# results has independent variables and dependent variables
def reuslt_handle(X_predict, y_predict, predict_result_address):
    y_predict = pd.DataFrame(y_predict)
    y_predict.columns = ['Species']
    predict_data = pd.concat([X_predict, y_predict], axis=1)
    predict_data.to_csv(predict_result_address, index=0)


if __name__ == '__main__':
    try:
        # a = []
        # for i in range(1, len(sys.argv)):
        #     # a.append((int(sys.argv[i]))), caution 'int' or 'str'
        #     a.append((str(sys.argv[i])))

        a = ["G:\Coding Program\General Algorithm\iris_train_classification.csv",
             "G:\Coding Program\General Algorithm\iris_test_classification.csv",
             "G:\Coding Program\General Algorithm\SVM_results.csv"]

        X_train, y_train, X_predict = get_data(a[0], a[1])

        # predict
        y_predict = prdedict_mode(X_train, y_train, X_predict)
        # output results to csv
        reuslt_handle(X_predict, y_predict, a[2])
        print_result = "1"
    except:
        print_result = "0"

    print(print_result)
