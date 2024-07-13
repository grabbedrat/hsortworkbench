import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf_embedding(texts):
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(texts).toarray()

def dummy_embedding(texts):
    # This is a placeholder for future embedding methods
    return np.random.rand(len(texts), 100)

EMBEDDING_METHODS = {
    'TF-IDF': tfidf_embedding,
    'Dummy': dummy_embedding,
}

def generate_embeddings(texts, method='TF-IDF'):
    if method not in EMBEDDING_METHODS:
        raise ValueError(f"Unknown embedding method: {method}")
    return EMBEDDING_METHODS[method](texts)