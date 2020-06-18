#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'Naive Bayes Classification'
__author__ = 'Fan Shuai'
__mtime__ = '2020/6/18'
"""


from sklearn.naive_bayes import MultinomialNB
from SVM_classification import get_data, reuslt_handle


"""
MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
"""


def prdedict_mode(X_train, y_train, X_predict):
    mnb = MultinomialNB()
    mnb.fit(X_train, y_train)
    y_predict = mnb.predict(X_predict)
    # print(y_predict)
    return y_predict


if __name__ == '__main__':
    try:
        # a = []
        # for i in range(1, len(sys.argv)):
        #     # a.append((int(sys.argv[i]))), caution 'int' or 'str'
        #     a.append((str(sys.argv[i])))

        a = ["G:\Coding Program\General Algorithm\iris_train.csv",
             "G:\Coding Program\General Algorithm\iris_test.csv",
             r"G:\Coding Program\General Algorithm\NB_results.csv"]

        X_train, y_train, X_predict = get_data(a[0], a[1])

        # predict
        y_predict = prdedict_mode(X_train, y_train, X_predict)
        # output results to csv
        reuslt_handle(X_predict, y_predict, a[2])
        print_result = "1"
    except:
        print_result = "0"

    print(print_result)
