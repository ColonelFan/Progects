#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'DBSCAN Cluster'
__author__ = 'Fan Shuai'
__mtime__ = '2020/6/30'
"""


from K_means_cluster import get_data_cl, reuslt_handle
from sklearn.cluster import DBSCAN


"""
DBSCAN(algorithm='auto', eps=0.4, leaf_size=30, metric='euclidean',
    metric_params=None, min_samples=9, n_jobs=1, p=None)
"""


def cluster_mode(X_data):
    dbscan = DBSCAN(eps=0.4, min_samples=9)
    dbscan.fit(X_data)

    # judge cluster result of predict data
    y_labels = dbscan.labels_

    # cluster of every sample
    # y_predict = clf.labels_
    # center of every cluster
    # cluster_centers = clf.cluster_centers_
    # print(y_predict)
    # print(cluster_centers)

    return y_labels


if __name__ == '__main__':
    try:
        # a = []
        # for i in range(1, len(sys.argv)):
        #     # a.append((int(sys.argv[i]))), caution 'int' or 'str'
        #     a.append((str(sys.argv[i])))

        a = ["G:\Coding Program\General Algorithm\iris_cluster.csv",
             "G:\Coding Program\General Algorithm\DBSCAN_results.csv"]

        X_data = get_data_cl(a[0])

        # cluster
        y_labels = cluster_mode(X_data)
        # output results to csv
        reuslt_handle(X_data, y_labels, a[1])
        print_result = "1"
    except:
        print_result = "0"

    print(print_result)
