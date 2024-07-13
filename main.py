import streamlit as st
import pandas as pd
import numpy as np
from data_loader import load_preprocessed_data, save_preprocessed_data
from embedding import generate_embeddings, EMBEDDING_METHODS
from clustering import perform_clustering, CLUSTERING_METHODS
from visualization import visualize_clusters, plot_silhouette_scores, plot_minimum_spanning_tree

def main():
    st.set_page_config(layout="wide", page_title="Hierarchical Sorting Workbench")
    st.title("Hierarchical Sorting Workbench")

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

    # Clustering
    st.header("2. Clustering")
    if 'embeddings' in st.session_state:
        if st.button("Perform All Clustering Methods"):
            for method_name, method in CLUSTERING_METHODS.items():
                with st.spinner(f"Performing {method_name} clustering..."):
                    params = method.get_params()
                    labels, silhouette_avg, mst = perform_clustering(st.session_state['embeddings'], method_name, **params)
                st.success(f"{method_name} clustering complete! Silhouette Score: {silhouette_avg:.4f}")
                st.session_state[f'{method_name}_labels'] = labels
                st.session_state[f'{method_name}_mst'] = mst
                st.session_state[f'{method_name}_silhouette'] = silhouette_avg
    else:
        st.warning("Please generate embeddings first.")

    # Visualization
    st.header("3. Visualization")
    col1, col2, col3 = st.columns(3)

    for i, method_name in enumerate(CLUSTERING_METHODS.keys()):
        with [col1, col2, col3][i]:
            st.subheader(f"{method_name} Visualization")
            if f'{method_name}_labels' in st.session_state and f'{method_name}_mst' in st.session_state:
                st.write(f"Silhouette Score: {st.session_state[f'{method_name}_silhouette']:.4f}")
                visualize_clusters(st.session_state['embeddings'], st.session_state[f'{method_name}_labels'], bookmarks_df)
                plot_silhouette_scores(st.session_state['embeddings'], st.session_state[f'{method_name}_labels'])
                plot_minimum_spanning_tree(st.session_state[f'{method_name}_mst'])
            else:
                st.info(f"Perform clustering to see {method_name} visualizations.")

if __name__ == "__main__":
    main()