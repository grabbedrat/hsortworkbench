import streamlit as st
import pandas as pd
import numpy as np
from data_loader import load_preprocessed_data, save_preprocessed_data
from embedding import generate_embeddings, EMBEDDING_METHODS
from clustering import perform_clustering, CLUSTERING_METHODS
from visualization import plot_folder_structure, display_folder_contents, plot_treemap

def main():
    st.set_page_config(layout="wide", page_title="Bookmark Folder Structure Generator")
    st.title("Bookmark Folder Structure Generator")

    # Load data
    bookmarks_df, _ = load_preprocessed_data()
    if bookmarks_df is None:
        st.error("Failed to load preprocessed data. Please check the data file.")
        return

    # Display data
    st.header("Data")
    st.write(bookmarks_df)

    # Embedding
    st.header("Embedding")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Method 1")
        embedding_method1 = st.selectbox("Embedding Method", list(EMBEDDING_METHODS.keys()), key='embedding_method1')
        if st.button("Generate Embeddings", key='generate_embeddings1'):
            with st.spinner("Generating embeddings..."):
                texts = (bookmarks_df['title'] + " " + bookmarks_df['url'] + " " + bookmarks_df['tags']).tolist()
                embeddings1 = generate_embeddings(texts, method=embedding_method1)
            st.success("Embeddings generated successfully!")
            save_preprocessed_data(bookmarks_df, embeddings1)
            st.session_state['embeddings1'] = embeddings1

    with col2:
        st.subheader("Method 2")
        embedding_method2 = st.selectbox("Embedding Method", list(EMBEDDING_METHODS.keys()), key='embedding_method2')
        if st.button("Generate Embeddings", key='generate_embeddings2'):
            with st.spinner("Generating embeddings..."):
                texts = (bookmarks_df['title'] + " " + bookmarks_df['url'] + " " + bookmarks_df['tags']).tolist()
                embeddings2 = generate_embeddings(texts, method=embedding_method2)
            st.success("Embeddings generated successfully!")
            save_preprocessed_data(bookmarks_df, embeddings2)
            st.session_state['embeddings2'] = embeddings2

    # Clustering and Visualization
    if 'embeddings1' in st.session_state and 'embeddings2' in st.session_state:
        st.header("Clustering and Visualization")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Method 1")
            clustering_method1 = st.selectbox("Clustering Method", list(CLUSTERING_METHODS.keys()), key='clustering_method1')
            params1 = CLUSTERING_METHODS[clustering_method1].get_params(key='params1')
            if st.button("Generate Folder Structure", key='generate_structure1'):
                with st.spinner("Generating folder structure..."):
                    hierarchy1, n_clusters1, silhouette_avg1 = perform_clustering(st.session_state['embeddings1'], clustering_method1, **params1)
                st.success(f"Folder structure generated with {n_clusters1} top-level folders! Silhouette Score: {silhouette_avg1:.4f}")
                st.session_state['hierarchy1'] = hierarchy1
            if 'hierarchy1' in st.session_state:
                plot_folder_structure(st.session_state['hierarchy1'], bookmarks_df)
                plot_treemap(st.session_state['hierarchy1'], bookmarks_df)
                display_folder_contents(st.session_state['hierarchy1'], bookmarks_df)
        
        with col2:
            st.subheader("Method 2")
            clustering_method2 = st.selectbox("Clustering Method", list(CLUSTERING_METHODS.keys()), key='clustering_method2')
            params2 = CLUSTERING_METHODS[clustering_method2].get_params(key='params2')
            if st.button("Generate Folder Structure", key='generate_structure2'):
                with st.spinner("Generating folder structure..."):
                    hierarchy2, n_clusters2, silhouette_avg2 = perform_clustering(st.session_state['embeddings2'], clustering_method2, **params2)
                st.success(f"Folder structure generated with {n_clusters2} top-level folders! Silhouette Score: {silhouette_avg2:.4f}")
                st.session_state['hierarchy2'] = hierarchy2
            if 'hierarchy2' in st.session_state:
                plot_folder_structure(st.session_state['hierarchy2'], bookmarks_df)
                plot_treemap(st.session_state['hierarchy2'], bookmarks_df)
                display_folder_contents(st.session_state['hierarchy2'], bookmarks_df)

if __name__ == "__main__":
    main()