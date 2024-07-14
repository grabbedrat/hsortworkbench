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
        'n_clusters': None,  # Set to None to get the full tree
        'distance_threshold': st.slider("Distance threshold", 0.0, 5.0, 0.5, 0.1, key=f"{key_prefix}_distance_threshold"),
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
        clusterer.fit(embeddings)
        hierarchy = create_agglomerative_hierarchy(clusterer, embeddings)
    else:  # HDBSCAN
        clusterer.fit(embeddings)
        hierarchy = create_hdbscan_hierarchy(clusterer, embeddings)
    
    flat_labels = get_flat_labels(hierarchy)
    n_clusters = len(set(flat_labels)) - (1 if -1 in flat_labels else 0)
    
    # Check if silhouette score can be calculated
    if n_clusters <= 1 or n_clusters >= len(embeddings):
        silhouette_avg = 0
        st.warning(f"Silhouette score cannot be calculated. Number of clusters: {n_clusters}, Number of samples: {len(embeddings)}")
    else:
        silhouette_avg = silhouette_score(embeddings, flat_labels)
    
    return hierarchy, n_clusters, silhouette_avg

def create_agglomerative_hierarchy(clusterer, embeddings):
    n_samples = len(embeddings)
    children = clusterer.children_
    distances = clusterer.distances_
    
    hierarchy = [{} for _ in range(n_samples * 2 - 1)]
    for i in range(n_samples):
        hierarchy[i] = {'node_id': i, 'size': 1, 'children': []}
    
    for i, (child1, child2) in enumerate(children):
        node_id = i + n_samples
        hierarchy[node_id] = {
            'node_id': node_id,
            'size': hierarchy[child1]['size'] + hierarchy[child2]['size'],
            'children': [hierarchy[child1], hierarchy[child2]],
            'distance': distances[i]
        }
    
    print(f"Agglomerative hierarchy created with {len(hierarchy)} nodes")
    print(f"Root node has {len(hierarchy[-1]['children'])} children")
    return hierarchy[-1]  # Return the root of the hierarchy

def create_hdbscan_hierarchy(clusterer, embeddings):
    labels = clusterer.labels_
    
    # Create a basic hierarchy with all points at the root
    hierarchy = {
        'node_id': 'root',
        'size': len(embeddings),
        'children': []
    }
    
    # Group points by cluster
    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(i)
    
    # Create child nodes for each cluster
    for label, points in clusters.items():
        child = {
            'node_id': f'cluster_{label}',
            'size': len(points),
            'children': [{'node_id': i, 'size': 1} for i in points]
        }
        hierarchy['children'].append(child)
    
    return hierarchy

def get_flat_labels(hierarchy):
    if 'label' in hierarchy:
        return [hierarchy['label']]
    if 'children' not in hierarchy:
        return [hierarchy.get('node_id', -1)]  # Return node_id or -1 if not found
    labels = []
    for child in hierarchy['children']:
        labels.extend(get_flat_labels(child))
    return labels