import streamlit as st
import pandas as pd
import numpy as np
from data_loader import load_preprocessed_data, save_preprocessed_data
from embedding import generate_embeddings, EMBEDDING_METHODS
from clustering import perform_clustering, CLUSTERING_METHODS
from visualization import plot_folder_structure, display_folder_contents

def main():
    st.set_page_config(layout="wide", page_title="Bookmark Folder Structure Generator")
    st.title("Bookmark Folder Structure Generator")

    # Load data
    bookmarks_df, _ = load_preprocessed_data()
    if bookmarks_df is None:
        st.error("Failed to load preprocessed data. Please check the data file.")
        return

    # Embedding
    st.header("1. Embedding")
    embedding_method = st.selectbox("Embedding Method", list(EMBEDDING_METHODS.keys()))
    if st.button("Generate Embeddings"):
        with st.spinner("Generating embeddings..."):
            texts = bookmarks_df['title'] + " " + bookmarks_df['url']
            embeddings = generate_embeddings(texts, method=embedding_method)
        st.success("Embeddings generated successfully!")
        save_preprocessed_data(bookmarks_df, embeddings)
        st.session_state['embeddings'] = embeddings

    # Clustering and Visualization
    if 'embeddings' in st.session_state:
        st.header("2. Generate Folder Structure")
        
        clustering_method = st.selectbox("Clustering Method", list(CLUSTERING_METHODS.keys()))
        st.write(CLUSTERING_METHODS[clustering_method].description)
        
        params = CLUSTERING_METHODS[clustering_method].get_params()

        if st.button("Generate Folder Structure"):
            with st.spinner("Generating folder structure..."):
                hierarchy, n_clusters, silhouette_avg = perform_clustering(st.session_state['embeddings'], clustering_method, **params)
            st.success(f"Folder structure generated with {n_clusters} top-level folders! Silhouette Score: {silhouette_avg:.4f}")
            st.session_state['hierarchy'] = hierarchy
            st.session_state['n_clusters'] = n_clusters
            st.session_state['silhouette_avg'] = silhouette_avg

        if 'hierarchy' in st.session_state:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Folder Structure Visualization")
                plot_folder_structure(st.session_state['hierarchy'], bookmarks_df)
            with col2:
                st.subheader("Folder Contents")
                display_folder_contents(st.session_state['hierarchy'], bookmarks_df)
    else:
        st.warning("Please generate embeddings first.")

if __name__ == "__main__":
    main()