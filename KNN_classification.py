#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'KNN Classification'
__author__ = 'Fan Shuai'
__mtime__ = '2020/6/18'
"""


from sklearn.neighbors import KNeighborsClassifier
from SVM_classification import get_data, reuslt_handle
from sklearn.preprocessing import StandardScaler


"""
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=5, p=2,
           weights='uniform')
"""


def prdedict_mode(X_train, y_train, X_predict):
    # normalize
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_predict = ss.transform(X_predict)

    # train and predict
    knc = KNeighborsClassifier()
    knc.fit(X_train, y_train)
    y_predict = knc.predict(X_predict)
    # print(y_predict)
    return y_predict


if __name__ == '__main__':
    try:
        # a = []
        # for i in range(1, len(sys.argv)):
        #     # a.append((int(sys.argv[i]))), caution 'int' or 'str'
        #     a.append((str(sys.argv[i])))

        a = ["G:\Coding Program\General Algorithm\iris_train_classification.csv",
             "G:\Coding Program\General Algorithm\iris_test_classification.csv",
             "G:\Coding Program\General Algorithm\KNN_results.csv"]

        X_train, y_train, X_predict = get_data(a[0], a[1])

        # predict
        y_predict = prdedict_mode(X_train, y_train, X_predict)
        # output results to csv
        reuslt_handle(X_predict, y_predict, a[2])
        print_result = "1"
    except:
        print_result = "0"

    print(print_result)
