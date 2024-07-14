import numpy as np
from sklearn.cluster import AgglomerativeClustering
from hdbscan import HDBSCAN
from sklearn.metrics import silhouette_score
import streamlit as st
from gensim import corpora
from gensim.models import LdaModel
from gensim.parsing.preprocessing import STOPWORDS
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

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

def get_lda_params(key_prefix=''):
    return {
        'num_topics': st.slider("Number of topics", 2, 50, 10, key=f"{key_prefix}_num_topics"),
        'passes': st.slider("Number of passes", 1, 20, 5, key=f"{key_prefix}_passes"),
        'chunksize': st.slider("Chunk size", 100, 5000, 2000, 100, key=f"{key_prefix}_chunksize"),
        'alpha': st.select_slider("Alpha", options=['symmetric', 'asymmetric', 'auto'], value='symmetric', key=f"{key_prefix}_alpha"),
        'eta': st.number_input("Eta", 0.001, 1.0, 0.1, 0.001, key=f"{key_prefix}_eta")
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
    ),
    'LDA': ClusteringMethod(
        'LDA',
        LdaModel,
        get_lda_params,
        "Latent Dirichlet Allocation for topic modeling. Discovers abstract topics in a collection of documents."
    )
}

def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text.lower())
    return [lemmatizer.lemmatize(token) for token in tokens if token not in STOPWORDS and token.isalnum()]

def perform_clustering(bookmarks_df, method, **params):
    if method not in CLUSTERING_METHODS:
        raise ValueError(f"Unknown clustering method: {method}")
    
    if method == 'LDA':
        return perform_lda_clustering(bookmarks_df, **params)
    else:
        embeddings = st.session_state['active_embeddings']
        clusterer = CLUSTERING_METHODS[method].algorithm(**params)
        
        if method == 'Agglomerative':
            clusterer.fit(embeddings)
            hierarchy = create_agglomerative_hierarchy(clusterer, embeddings)
        else:  # HDBSCAN
            clusterer.fit(embeddings)
            hierarchy = create_hdbscan_hierarchy(clusterer, embeddings)
        
        flat_labels = get_flat_labels(hierarchy)
        n_clusters = len(set(flat_labels)) - (1 if -1 in flat_labels else 0)
        
        if n_clusters <= 1 or n_clusters >= len(embeddings):
            silhouette_avg = 0
            st.warning(f"Silhouette score cannot be calculated. Number of clusters: {n_clusters}, Number of samples: {len(embeddings)}")
        else:
            silhouette_avg = silhouette_score(embeddings, flat_labels)
        
        return hierarchy, n_clusters, silhouette_avg

def perform_lda_clustering(bookmarks_df, num_topics, passes, chunksize, alpha, eta):
    # Combine title, URL, and tags for each bookmark
    texts = (bookmarks_df['title'] + " " + bookmarks_df['url'] + " " + bookmarks_df['tags']).tolist()
    
    # Preprocess the texts
    processed_texts = [preprocess_text(text) for text in texts]
    
    # Create a dictionary and corpus
    dictionary = corpora.Dictionary(processed_texts)
    corpus = [dictionary.doc2bow(text) for text in processed_texts]
    
    # Train the LDA model
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=passes,
        chunksize=chunksize,
        alpha=alpha,
        eta=eta
    )
    
    # Create hierarchy based on LDA results
    hierarchy = create_lda_hierarchy(lda_model, corpus, bookmarks_df)
    
    n_clusters = num_topics
    silhouette_avg = 0  # LDA doesn't have a straightforward silhouette score
    
    return hierarchy, n_clusters, silhouette_avg

def create_lda_hierarchy(lda_model, corpus, bookmarks_df):
    # Create root node
    hierarchy = {
        'node_id': 'root',
        'size': len(bookmarks_df),
        'children': []
    }
    
    # Create a topic for each LDA topic
    for topic_id in range(lda_model.num_topics):
        topic_node = {
            'node_id': f'topic_{topic_id}',
            'size': 0,
            'children': []
        }
        hierarchy['children'].append(topic_node)
    
    # Assign documents to topics
    for doc_id, doc_topics in enumerate(lda_model[corpus]):
        best_topic = max(doc_topics, key=lambda x: x[1])
        topic_id, _ = best_topic
        
        bookmark_node = {
            'node_id': doc_id,
            'size': 1
        }
        hierarchy['children'][topic_id]['children'].append(bookmark_node)
        hierarchy['children'][topic_id]['size'] += 1
    
    return hierarchy

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