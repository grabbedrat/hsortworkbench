import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import hdbscan
import streamlit as st
from sklearn.neighbors import kneighbors_graph

class ClusteringMethod:
    def __init__(self, name, algorithm, param_func):
        self.name = name
        self.algorithm = algorithm
        self.param_func = param_func

    def get_params(self):
        return self.param_func()

def get_kmeans_params():
    return {
        'n_clusters': st.slider("Number of clusters (K-Means)", 2, 20, 5),
        'random_state': 42
    }

def get_agglomerative_params():
    return {
        'n_clusters': st.slider("Number of clusters (Agglomerative)", 2, 20, 5),
        'linkage': st.selectbox("Linkage", ['ward', 'complete', 'average', 'single'])
    }

def get_hdbscan_params():
    return {
        'min_cluster_size': st.slider("Min cluster size", 2, 20, 5),
        'min_samples': st.slider("Min samples", 1, 10, 3),
        'cluster_selection_epsilon': st.slider("Cluster selection epsilon", 0.0, 1.0, 0.0, 0.1)
    }

CLUSTERING_METHODS = {
    'K-Means': ClusteringMethod('K-Means', KMeans, get_kmeans_params),
    'Agglomerative': ClusteringMethod('Agglomerative', AgglomerativeClustering, get_agglomerative_params),
    'HDBSCAN': ClusteringMethod('HDBSCAN', hdbscan.HDBSCAN, get_hdbscan_params)
}

def perform_clustering(embeddings, method, **params):
    if method not in CLUSTERING_METHODS:
        raise ValueError(f"Unknown clustering method: {method}")
    
    clusterer = CLUSTERING_METHODS[method].algorithm(**params)
    
    if method == 'HDBSCAN':
        clusterer.fit(embeddings)
        labels = clusterer.labels_
        try:
            mst = clusterer.minimum_spanning_tree_
        except AttributeError:
            # If MST is not available, create a k-nearest neighbors graph
            mst = kneighbors_graph(embeddings, n_neighbors=5, mode='distance')
    else:
        labels = clusterer.fit_predict(embeddings)
        # For non-HDBSCAN methods, we'll use a k-nearest neighbors graph
        mst = kneighbors_graph(embeddings, n_neighbors=5, mode='distance')
    
    silhouette_avg = silhouette_score(embeddings, labels)
    
    return labels, silhouette_avg, mst