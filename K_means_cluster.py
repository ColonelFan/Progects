#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'K-means Cluster'
__author__ = 'Fan Shuai'
__mtime__ = '2020/6/29'
"""


from sklearn.cluster import KMeans
import numpy as np
import pandas as pd


"""
KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
    n_clusters=10, n_init=10, n_jobs=1, precompute_distances='auto',
    random_state=None, tol=0.0001, verbose=0)
"""


def get_data_cl(data_address):
    data = pd.read_csv(data_address)
    X_data = data.iloc[:, 0:(data.shape[1] - 1)]
    y_data = data.iloc[:, (data.shape[1] - 1)]

    return X_data, y_data


def cluster_mode(X_data, n_clusters):
    # initial KMeans model，set cluster center amount
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X_data)
    # get cluster result
    y_labels = kmeans.labels_
    return y_labels


# results after cluster
def reuslt_handle(X_data, y_labels, cluster_result_address):
    X_data = pd.DataFrame(X_data)
    y_labels = pd.DataFrame(y_labels)
    y_labels.columns = ['Species']
    cluster_data = pd.concat([X_data, y_labels], axis=1)
    cluster_data.to_csv(cluster_result_address, index=0)


if __name__ == '__main__':
    try:
        # a = []
        # for i in range(1, len(sys.argv)):
        #     # a.append((int(sys.argv[i]))), caution 'int' or 'str'
        #     a.append((str(sys.argv[i])))

        a = ["G:\Coding Program\General Algorithm\iris_train.csv",
             "G:\Coding Program\General Algorithm\K_means_results.csv"]

        X_data, y_data = get_data_cl(a[0])
        # cluster center amount, equal to labels amount
        n_clusters = len(np.unique(y_data))

        # cluster
        y_labels = cluster_mode(X_data, n_clusters)
        # output results to csv
        reuslt_handle(X_data, y_labels, a[1])
        print_result = "1"
    except:
        print_result = "0"

    print(print_result)
