import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tqdm import tqdm

def cluster_encodings(df, col, n_clusters):
    """
    Clusters the encodings given by the column `col` in the input dataframe.
    """
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(df[col].tolist())
    return kmeans

def plot_elbow_of_encoding_cluster(df, col, k_max):
    """
    Plots elbow curve of clustering up to k_max clusters.
    """
    distorsions = []
    for k in tqdm(range(2, k_max), desc=f"Running kmeans from k = 2 to {k_max}"):
        kmeans = cluster_encodings(df, col, k)
        distorsions.append(kmeans.inertia_)

    fig = plt.figure(figsize=(15, 5))
    plt.plot(range(2, k_max), distorsions)
    plt.grid(True)
    plt.title('Elbow curve')

def plot_num_elements_per_cluster(df, col, n_clusters):
    kmeans = cluster_encodings(df, col, n_clusters)
    counts = pd.DataFrame(kmeans.labels_).value_counts()
    counts = counts.reset_index()
    counts.columns = ['cluster', 'count']
    plt.bar(x=counts['cluster'], height=counts['count'])
    plt.title('Number of elements per cluster for k = ' + str(n_clusters))
    plt.xlabel('Cluster')
    plt.ylabel('Number of elements')
    plt.show()