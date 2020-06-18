#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'RandForest Classification'
__author__ = 'Fan Shuai'
__mtime__ = '2020/6/17'
"""


from sklearn.ensemble import RandomForestClassifier
from SVM_classification import get_data, reuslt_handle


"""
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
"""


def prdedict_mode(X_train, y_train, X_predict):
    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)
    y_predict = rfc.predict(X_predict)
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
             "G:\Coding Program\General Algorithm\RF_results.csv"]

        X_train, y_train, X_predict = get_data(a[0], a[1])

        # predict
        y_predict = prdedict_mode(X_train, y_train, X_predict)
        # output results to csv
        reuslt_handle(X_predict, y_predict, a[2])
        print_result = "1"
    except:
        print_result = "0"

    print(print_result)
