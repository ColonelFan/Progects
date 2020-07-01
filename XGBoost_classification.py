#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'Fan Shuai'
__mtime__ = '2020/6/17'
"""


from xgboost import XGBClassifier
from SVM_classification import get_data, reuslt_handle
from sklearn.preprocessing import StandardScaler


"""
XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
       importance_type='gain', interaction_constraints='',
       learning_rate=0.300000012, max_delta_step=0, max_depth=6,
       min_child_weight=1, missing=nan, monotone_constraints='()',
       n_estimators=100, n_jobs=0, num_parallel_tree=1,
       objective='multi:softprob', random_state=0, reg_alpha=0,
       reg_lambda=1, scale_pos_weight=None, subsample=1,
       tree_method='exact', validate_parameters=1, verbosity=None)
"""


def prdedict_mode(X_train, y_train, X_predict):
    # normalize
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_predict = ss.transform(X_predict)

    xgbc = XGBClassifier()
    xgbc.fit(X_train, y_train)
    y_predict = xgbc.predict(X_predict)
    return y_predict


if __name__ == '__main__':
    try:
        # a = []
        # for i in range(1, len(sys.argv)):
        #     # a.append((int(sys.argv[i]))), caution 'int' or 'str'
        #     a.append((str(sys.argv[i])))

        a = ["G:\Coding Program\General Algorithm\iris_train_classification.csv",
             "G:\Coding Program\General Algorithm\iris_test_classification.csv",
             "G:\Coding Program\General Algorithm\XGBoost_results.csv"]

        X_train, y_train, X_predict = get_data(a[0], a[1])

        # predict
        y_predict = prdedict_mode(X_train, y_train, X_predict)
        # output results to csv
        reuslt_handle(X_predict, y_predict, a[2])
        print_result = "1"
    except:
        print_result = "0"

    print(print_result)
