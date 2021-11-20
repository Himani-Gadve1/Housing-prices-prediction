#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 22:07:44 2021

@author: himani
"""

# Import all required libraries
import numpy as np
import pandas as pd
import gower
from pyclustering.cluster.kmedoids import kmedoids
from sklearn.metrics.cluster import normalized_mutual_info_score

# In[3]:
# Load data set
data = pd.read_csv('/Users/himani/Documents/Fall 2021 SFSU/DS862 projects/project 1/train.csv')
data = data.drop('Id', axis = 1)

# Remove columns that have too many missing values
data = data.drop(data.columns[data.isnull().sum() > 30], axis = 1)

# Remove missing values
data.dropna(inplace = True)
print(data.head())

# In[3]:

X = data.copy()
del X['SalePrice']
y = data['SalePrice']
X.shape # (1451, 63)
y.shape # (1451,)

# 2.Compute the Gower distance of the full predictors set, i.e. no train/test split
gd_matrix = gower.gower_matrix(X)
gd_matrix

# In[3]:

# To use the K-medoid function, provide some initial centers. Let's take k =5.
# Randomly sample 5 observations as centers.
np.random.seed(50)
k = 5
center_index = np.random.randint(0, len(y), k)
print("centers: ",center_index)

# 3. Apply K-medoids using the gower distance matrix as input.
kmedoids_gd = kmedoids(gd_matrix, center_index, data_type='distance_matrix')

# Run cluster analysis and obtain results
kmedoids_gd.process()
# finding new medoids
medoids = kmedoids_gd.get_medoids()
# finding new clusters
clusters = kmedoids_gd.get_clusters()

print("Medoids:",medoids)

# Cluster output for each cluster of size 5
for i in range(k):
    print(f"cluster in {i} is {clusters[i]}")
    print(clusters[i])



#result:
#clustering result tells us which observations belong to cluster k

# In[3]:
# 4.a first create an array that records the cluster membership of each observation
# Assign labels to Clusters
labels = np.zeros([len(X)], dtype=int)
for i in range(k):
    labels[clusters[i]] = i
#    print(labels)

for i in range(5):
    print(y[labels==i].mean())

#labels
#Categories (5, int64): [0 < 1 < 2 < 3 < 4]
#146266.26962457338
#130985.00852272728
#238547.92651757188
#236582.3855799373
#132112.3563218391




# In[3]:

# 4.b Bin the response variable (of the original data set) into the number of categories you used for k-medoids
bins = pd.qcut(y, k, range(k))
print('response variable binned into 5 categories :',bins)
for i in range(5):
    print(y[bins==i].mean())



#bins
# Categories (5, int64): [0 < 1 < 2 < 3 < 4]
#100760.70169491526
#135884.31632653062
#163317.5809859155
#201066.3676975945
#304943.1010452962

#Observation:

#means for each cluster using kmedoids is non sequential whereas mean of each cluster using qcut is sequential

# In[3]:

# 4. c Compute the normalized mutual information (NMI) between your clustering results and the binned categories.
NMI_5 = normalized_mutual_info_score(bins, labels)
print("NMI score for K = 5 is", NMI_5)
#NMI score for K = 5 is 0.21221029395430538

# In[3]:
#Let's compute NMI for several K values to compare the results

# Randomly sample 4 observations as initial centers.
np.random.seed(50)
k = 4
center_index = np.random.randint(0, len(y), k)
print("centers: ",center_index)

# 3. Apply K-medoids using the gower distance matrix as input.
kmedoids_gd = kmedoids(gd_matrix, center_index, data_type='distance_matrix')

# Run cluster analysis and obtain results
kmedoids_gd.process()
# finding new medoids
medoids = kmedoids_gd.get_medoids()
# finding new clusters
clusters = kmedoids_gd.get_clusters()

print("Medoids:",medoids)

# Cluster output for each cluster of size 5
for i in range(k):
    print(f"cluster in {i} is {clusters[i]}")

# 4.a first create an array that records the cluster membership of each observation
# Assign labels to Clusters
labels = np.zeros([len(X)], dtype=int)
for i in range(k):
    labels[clusters[i]] = i
    print(labels)

# 4.b Bin the response variable (of the original data set) into the number of categories you used for k-medoids
bins = pd.qcut(y, k, range(k))
print('response variable binned into 4 categories :',bins)

# 4. c Compute the normalized mutual information (NMI) between your clustering results and the binned categories.
NMI_4 = normalized_mutual_info_score(bins, labels)
print("NMI score for K = 4 is", NMI_4)
#NMI score for K = 4 is 0.24005278162951577

# In[3]:
# for cluster k = 3
# Randomly sample 3 observations as initial centers.
np.random.seed(50)
k = 3
center_index = np.random.randint(0, len(y), k)
print("centers: ",center_index)

# 3. Apply K-medoids using the gower distance matrix as input.
kmedoids_gd = kmedoids(gd_matrix, center_index, data_type='distance_matrix')

# Run cluster analysis and obtain results
kmedoids_gd.process()
# finding new medoids
medoids = kmedoids_gd.get_medoids()
# finding new clusters
clusters = kmedoids_gd.get_clusters()

print("Medoids:",medoids)

# Cluster output for each cluster of size 3
for i in range(k):
    print(f"cluster in {i} is {clusters[i]}")

# 4.a first create an array that records the cluster membership of each observation
# Assign labels to Clusters
labels = np.zeros([len(X)], dtype=int)
for i in range(k):
    labels[clusters[i]] = i
    print(labels)

# 4.b Bin the response variable (of the original data set) into the number of categories you used for k-medoids
bins = pd.qcut(y, k, range(k))
print('response variable binned into 3 categories :',bins)

# 4. c Compute the normalized mutual information (NMI) between your clustering results and the binned categories.
NMI_3 = normalized_mutual_info_score(bins, labels)
print("NMI score for K = 3 is", NMI_3)
#NMI score for K = 3 is 0.28863032518077514

# In[3]:

# for cluster k = 2
# Randomly sample 2 observations as initial centers.
np.random.seed(50)
k = 2
center_index = np.random.randint(0, len(y), k)
print("centers: ",center_index)

# 3. Apply K-medoids using the gower distance matrix as input.
kmedoids_gd = kmedoids(gd_matrix, center_index, data_type='distance_matrix')

# Run cluster analysis and obtain results
kmedoids_gd.process()
# finding new medoids
medoids = kmedoids_gd.get_medoids()
# finding new clusters
clusters = kmedoids_gd.get_clusters()

print("Medoids:",medoids)

# Cluster output for each cluster of size 2
for i in range(k):
    print(f"cluster in {i} is {clusters[i]}")

# 4.a first create an array that records the cluster membership of each observation
# Assign labels to Clusters
labels = np.zeros([len(X)], dtype=int)
for i in range(k):
    labels[clusters[i]] = i
    print(labels)

# 4.b Bin the response variable (of the original data set) into the number of categories you used for k-medoids
bins = pd.qcut(y, k, range(k))
print('response variable binned into 2 categories :',bins)

# 4. c Compute the normalized mutual information (NMI) between your clustering results and the binned categories.
NMI_2 = normalized_mutual_info_score(bins, labels)
print("NMI score for K = 2 is", NMI_2)
#NMI score for K = 2 is 0.39546405815174285

#Observation:
#Using NMI score we are determining the score of clustering
#NMI score increases as the number of cluster decreases. NMI score is highest for clustter size of 2 and NMI score is lowest for cluster size as 5
# In[3]:
