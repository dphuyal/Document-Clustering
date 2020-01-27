#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: phuyaldeep

sources cited:
    https://masongallo.github.io/machine/learning,/python/2016/07/29/cosine-similarity.html
    Google and bunch of other websites
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

t=time.time()

df = pd.read_csv('doc2vec.csv', skiprows = 1, header = None)

# deletes first column
df = df.drop(df.columns[0], axis=1)

# drops NaN present across the dataset
df = df.dropna(how='any')
print('Df shape: ', df.shape)
df.head()
docs = df.as_matrix()

# randomly choose the k number of centroids from the given data set
def random_centroids(data, k):
    n_samples, n_features = np.shape(data)
    centroids = np.zeros((k, n_features))
    for i in range(k):
        centroid = data[np.random.choice(range(n_samples))]
        centroids[i] = centroid
    return centroids

# calculate Euclidean distance between two vectors
def euclidean_distance(a, b):
#    dist=0
#    for i in range(len(a)):
#        dist=dist + (a[i] - b[i])**2
#    distance = dist**(1/2)
#    return distance

    return np.sqrt(np.sum(np.square(a - b)))

# caluclate cosine similarity of two vectors
def cos_sim(a, b):
    dot_product = np.dot(a,b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

# return the index of the closest centroid to the sample
def closest_centroid(sample, centroids):
    closest_cent = 0
    closest_dist = float('inf') #99999999.99
    for i, centroid in enumerate(centroids):
        distance = cos_sim(sample, centroid)
        if distance < closest_dist:
            closest_cent = i
            closest_dist = distance
    return closest_cent

# assign the data points to its closest centroids to create clusters
def create_clusters(centroids, data, k):
    clusters = []
    for i in range(k):
        clusters.append([])
    for sample_i, sample in enumerate(data):
        centroid_i = closest_centroid(sample, centroids)
        clusters[centroid_i].append(sample_i)
    return clusters

# calculate new centroids as the mean of all the data points assigned to that centroid's cluster
def calculate_centroids(clusters, data, K):
    n_features = np.shape(data)[1]
    centroids = np.zeros((K, n_features))
    for i, cluster in enumerate(clusters):
        centroid = np.mean(data[cluster], axis=0)
        centroids[i] = centroid
    return centroids

# assign lables to the clusters
def cluster_labels(clusters, data):
    y_pred = np.zeros(np.shape(data)[0])
    for cluster_i, cluster in enumerate(clusters):
        for sample_i in cluster:
            y_pred[sample_i] = cluster_i
    return y_pred

# implementation of K-means clustering
def k_means(data, K, max_iterations):
    centroids = random_centroids(data, K)
    for i in range(max_iterations):
        clusters = create_clusters(centroids, data, K)
        prev_centroids = centroids # saves current centroids
        centroids = calculate_centroids(clusters, data, K) # calculates new centroids from clusters
        # if not centroids has changed
        diff = centroids - prev_centroids
        if not diff.any():
            break
        return cluster_labels(clusters, data), centroids, clusters

# caculate SSE. Sum of Square of distance between centroid and cluster's datapoints
def SSE(data, pred_cluster, pred_centroids, K):
    sse = 0
    for i in range(len(pred_cluster)):
        k = 0
        while k <= K:
            if pred_cluster[i] == k:
                sse += 1-cos_sim(data[i], pred_centroids[k])
                break
            k += 1
    return sse

# get SSE for cluster
sse={}

k_dict = {5: 100,
          7: 100,
          10: 100,
          12: 100,
          15: 100,
        }

np.random.seed(23)


for k in k_dict.keys():
    pred_labels, centroids, clusters = k_means(docs, k, k_dict[k])
    sse[k] = SSE(docs, pred_labels, centroids, k)
    print ('SSE for K = ', str(k), ' is ', sse[k])

plt.figure()
plt.plot(list(sse.keys()), list(sse.values()), '-o')
plt.xlabel('Number of Clusters')
plt.ylabel('SSE Score')
plt.title('SSE vs Number of Clusters for K-means')
plt.grid(True)
plt.show()

print("Done in  %0.3fs." % (time.time() - t))
