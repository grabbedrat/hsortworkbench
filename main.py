import streamlit as st
import pandas as pd
import numpy as np
import json
from data_loader import load_preprocessed_data, save_preprocessed_data
from sentence_transformers import SentenceTransformer
from clustering import perform_clustering, CLUSTERING_METHODS
from visualization import plot_folder_structure, display_folder_contents, plot_treemap, get_folder_name


def main():
    st.set_page_config(layout="wide", page_title="Bookmark Folder Structure Generator")
    st.title("Bookmark Folder Structure Generator")

    # Load data
    bookmarks_df, embeddings = load_preprocessed_data()
    if bookmarks_df is None:
        st.error("Failed to load preprocessed data. Please check the data file.")
        return

    # Display data in a collapsible section
    with st.expander("View Bookmarks Data", expanded=False):
        st.dataframe(bookmarks_df)

    # Embedding
    st.header("Embedding and Clustering")
    
    generate_embeddings(bookmarks_df, embeddings)
    
    # Clustering and Visualization
    if 'active_embeddings' in st.session_state:
        clustering_and_visualization(bookmarks_df)

def generate_embeddings(bookmarks_df, existing_embeddings):
    st.subheader("BERT Embedding")
    if existing_embeddings is not None:
        st.info("Embeddings already exist. You can regenerate them if needed.")
    
    if st.button("Generate BERT Embeddings"):
        texts = (bookmarks_df['title'] + " " + bookmarks_df['url'] + " " + bookmarks_df['tags']).tolist()
        
        # Create a progress bar
        progress_bar = st.progress(0)
        
        # Load BERT model
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        
        # Generate embeddings with progress updates
        embeddings = []
        for i, text in enumerate(texts):
            embedding = model.encode(text)
            embeddings.append(embedding)
            progress_bar.progress((i + 1) / len(texts))
        
        embeddings = np.array(embeddings)
        
        st.success("Embeddings generated successfully!")
        save_preprocessed_data(bookmarks_df, embeddings)
        st.session_state['active_embeddings'] = embeddings
        st.experimental_rerun()  # Rerun to update the clustering section
    elif existing_embeddings is not None:
        st.session_state['active_embeddings'] = existing_embeddings

def clustering_and_visualization(bookmarks_df):
    st.subheader("Clustering and Visualization")
    
    clustering_method = st.selectbox("Clustering Method", list(CLUSTERING_METHODS.keys()), key='clustering_method')
    params = CLUSTERING_METHODS[clustering_method].get_params(key_prefix='clustering')
    
    depth_limit = st.slider("Depth Limit", 1, 10, 5, key='depth_limit')  # Increased default to 5
    
    if st.button("Generate Folder Structure", key='generate_structure'):
        with st.spinner("Generating folder structure..."):
            if clustering_method == 'LDA':
                hierarchy, n_clusters, silhouette_avg = perform_clustering(bookmarks_df, clustering_method, **params)
            else:
                hierarchy, n_clusters, silhouette_avg = perform_clustering(st.session_state['active_embeddings'], clustering_method, **params)
        st.success(f"Folder structure generated with {n_clusters} top-level folders! Silhouette Score: {silhouette_avg:.4f}")
        st.session_state['hierarchy'] = hierarchy
        
        # Create a download button for the folder structure
        folder_structure = create_folder_structure_dict(hierarchy, bookmarks_df)
        json_structure = json.dumps(folder_structure, indent=2)
        st.download_button(
            label="Download Folder Structure",
            data=json_structure,
            file_name="folder_structure.json",
            mime="application/json"
        )
    
    if 'hierarchy' in st.session_state:
        st.subheader("Folder Structure Visualization")
        st.write(f"Hierarchy contains {count_nodes(st.session_state['hierarchy'])} nodes")
        plot_folder_structure(st.session_state['hierarchy'], bookmarks_df, depth_limit)
        
        st.subheader("Treemap Visualization")
        plot_treemap(st.session_state['hierarchy'], bookmarks_df)
        
        st.subheader("Folder Contents")
        display_folder_contents(st.session_state['hierarchy'], bookmarks_df)

def create_folder_structure_dict(hierarchy, bookmarks_df):
    def build_structure(node):
        if 'children' not in node:
            bookmark = bookmarks_df.iloc[node['node_id']]
            return {
                "title": bookmark['title'],
                "url": bookmark['url'],
                "tags": bookmark['tags']
            }
        else:
            folder_name = get_folder_name(node, bookmarks_df)
            return {
                folder_name: [build_structure(child) for child in node['children']]
            }
    
    return build_structure(hierarchy)

def count_nodes(node):
    if 'children' not in node:
        return 1
    return 1 + sum(count_nodes(child) for child in node['children'])

if __name__ == "__main__":
    main()