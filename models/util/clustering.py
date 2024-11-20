import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import davies_bouldin_score

def cluster_encodings(df, col, method="kmeans", config={}):
    """
    Clusters the encodings given by the column `col` in the input dataframe.
    """
    if method == "kmeans":
        if "n_clusters" not in config:
            raise ValueError("n_clusters not found in config")
        kmeans = KMeans(n_clusters=config["n_clusters"])
        kmeans.fit(df[col].tolist())
        return kmeans
    elif method == "dbscan":
        if "eps" not in config:
            raise ValueError("eps not found in config")
        if "min_samples" not in config:
            raise ValueError("min_samples not found in config")
        dbscan = DBSCAN(eps=config["eps"], min_samples=config["min_samples"])
        dbscan.fit(df[col].tolist())
        return dbscan

def plot_elbow_of_encoding_cluster(df, col, k_max):
    """
    Plots elbow curve of clustering up to k_max clusters.
    """
    distorsions = []
    for k in tqdm(range(2, k_max), desc=f"Running kmeans from k = 2 to {k_max}"):
        kmeans = cluster_encodings(df, col,  method="kmeans", config={"n_clusters": k})
        distorsions.append(kmeans.inertia_)

    fig = plt.figure(figsize=(15, 5))
    plt.plot(range(2, k_max), distorsions)
    plt.grid(True)
    plt.title('Elbow curve')
    plt.xlabel('Number of clusters')
    plt.ylabel('Intertia')

def get_sillhouette_score(kmeans, df, col):
    labels = kmeans.labels_
    silhouette_avg = silhouette_score(df[col].tolist(), labels)
    return silhouette_avg

def plot_num_clusters_vs_sillhouette(df, col, k_max):
    """
    Plots sillhouette score against number of clusters.

    Ideally, pick k such that sillhouette score is maximized.
    """
    scores = []
    for k in tqdm(range(2, k_max), desc=f"Running kmeans from k = 2 to {k_max}"):
        kmeans = cluster_encodings(df, col,  method="kmeans", config={"n_clusters": k})
        scores.append(get_sillhouette_score(kmeans, df, col))
    fig = plt.figure(figsize=(15, 5))
    plt.plot(range(2, k_max), scores)
    plt.grid(True)
    plt.title('Sillhouette scores')

def get_davies_bouldin_index(kmeans, df, col):
    labels = kmeans.labels_
    davies_bouldin_avg = davies_bouldin_score(df[col].tolist(), labels)
    return davies_bouldin_avg

def plot_num_clusters_vs_db_score(df, col, k_max):
    """
    Plots sillhouette score against number of clusters.

    Ideally, pick k such that davies-bouldin index is maximized.
    """
    scores = []
    for k in tqdm(range(2, k_max), desc=f"Running kmeans from k = 2 to {k_max}"):
        kmeans = cluster_encodings(df, col, method="kmeans", config={"n_clusters": k})
        scores.append(get_davies_bouldin_index(kmeans, df, col))
    fig = plt.figure(figsize=(15, 5))
    plt.plot(range(2, k_max), scores)
    plt.grid(True)
    plt.title('Davies-Bouldin Indices vs num clusters')

def plot_num_elements_per_cluster(df, col, n_clusters):
    kmeans = cluster_encodings(df, col, method="kmeans", config={"n_clusters": n_clusters})
    counts = pd.DataFrame(kmeans.labels_).value_counts()
    counts = counts.reset_index()
    counts.columns = ['cluster', 'count']
    plt.bar(x=counts['cluster'], height=counts['count'])
    plt.title('Number of elements per cluster for k = ' + str(n_clusters))
    plt.xlabel('Cluster')
    plt.ylabel('Number of elements')
    plt.show()