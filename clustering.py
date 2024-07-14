import numpy as np
from sklearn.cluster import AgglomerativeClustering
from hdbscan import HDBSCAN
from sklearn.metrics import silhouette_score
import streamlit as st

class ClusteringMethod:
    def __init__(self, name, algorithm, param_func, description):
        self.name = name
        self.algorithm = algorithm
        self.param_func = param_func
        self.description = description

    def get_params(self, key_prefix=''):
        return self.param_func(key_prefix)

def get_agglomerative_params(key_prefix=''):
    return {
        'n_clusters': st.slider("Number of clusters", 2, 20, 5, key=f"{key_prefix}_n_clusters"),
        'linkage': st.selectbox("Linkage", ['ward', 'complete', 'average', 'single'], key=f"{key_prefix}_linkage")
    }

def get_hdbscan_params(key_prefix=''):
    return {
        'min_cluster_size': st.slider("Minimum cluster size", 2, 20, 5, key=f"{key_prefix}_min_cluster_size"),
        'min_samples': st.slider("Minimum samples", 1, 10, 1, key=f"{key_prefix}_min_samples"),
        'cluster_selection_epsilon': st.slider("Cluster selection epsilon", 0.0, 1.0, 0.0, 0.1, key=f"{key_prefix}_cluster_selection_epsilon")
    }

CLUSTERING_METHODS = {
    'Agglomerative': ClusteringMethod(
        'Agglomerative', 
        AgglomerativeClustering, 
        get_agglomerative_params,
        "Builds a hierarchy of clusters from the bottom up. Good for creating a traditional folder structure."
    ),
    'HDBSCAN': ClusteringMethod(
        'HDBSCAN', 
        HDBSCAN, 
        get_hdbscan_params,
        "Density-based clustering that can find clusters of varying densities and shapes. Can identify 'noise' points."
    )
}

def perform_clustering(embeddings, method, **params):
    if method not in CLUSTERING_METHODS:
        raise ValueError(f"Unknown clustering method: {method}")
    
    clusterer = CLUSTERING_METHODS[method].algorithm(**params)
    
    if method == 'Agglomerative':
        labels = clusterer.fit_predict(embeddings)
    else:  # HDBSCAN
        clusterer.fit(embeddings)
        labels = clusterer.labels_
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # Account for noise in HDBSCAN
    
    # Create a hierarchical structure
    hierarchy = {}
    for i, label in enumerate(labels):
        if label not in hierarchy:
            hierarchy[label] = []
        hierarchy[label].append(i)
    
    silhouette_avg = silhouette_score(embeddings, labels) if n_clusters > 1 else 0
    
    return hierarchy, n_clusters, silhouette_avg