import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

def tfidf_embedding(texts):
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(texts).toarray()

def transformer_embedding(texts):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode(texts)

EMBEDDING_METHODS = {
    'TF-IDF': tfidf_embedding,
    'Transformer': transformer_embedding
}

def generate_embeddings(texts, method='TF-IDF'):
    if method not in EMBEDDING_METHODS:
        raise ValueError(f"Unknown embedding method: {method}")
    return EMBEDDING_METHODS[method](texts)