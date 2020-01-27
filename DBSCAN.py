#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: phuyaldeep

Sources cited: 
    https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html
"""
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN


df = pd.read_csv('doc2vec.csv', skiprows = 1, header = None)
# delete first column
df = df.drop(df.columns[0], axis=1)

df = df.dropna(how='any')
print('Df shape: ', df.shape)
    
# sample data
np.random.seed(0)
df = df.sample(n = 20000)

print('Shape: ', df.shape)
df.head()

docs = df.as_matrix()

db_scan = DBSCAN(eps=0.2, min_samples=6,metric='cosine').fit(docs)

labels = db_scan.labels_

n_clusters_ = len(set(labels)) - (1 if -1 else 0)
n_noise_ = list(labels).count(-1)

print('Labels shape: ', np.shape(labels))

cluster_dict = {}

for i in range(n_clusters_ + 1):
    cluster_dict[i - 1] = []
    
print(cluster_dict)

for i in range(len(db_scan.labels_)):
    cluster_dict[db_scan.labels_[i]].append(df.iloc[i])
    
noise_points = cluster_dict[-1]
print('Noise points shape: ', np.shape(noise_points))
   
for k in range(n_clusters_):
    cluster_points = cluster_dict[k] 
    print('Cluster', str(k), ' points shape: ', np.shape(cluster_points))
    
cluster_center_dict = {}

for i in range(n_clusters_):
    cluster_center_dict[i - 1] = np.mean(cluster_dict[i], axis = 0)
    #print('Cluster ', str(i), 'center: ', cluster_center_dict[i - 1])
    
#sse = 0
#print (cluster_center_dict.keys())
#for i in cluster_center_dict.keys():
#    while k <= n_clusters_:
#        if cluster_center_dict[i] == n_clusters_:
#            sse += euclidean_distance(data[i], cluster_center_dict[k])
#            break
#        k += 1


























 