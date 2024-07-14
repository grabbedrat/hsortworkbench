import streamlit as st
import pandas as pd
import numpy as np
import json
from data_loader import load_preprocessed_data, save_preprocessed_data
from embedding import generate_embeddings, EMBEDDING_METHODS
from clustering import perform_clustering, CLUSTERING_METHODS
from visualization import plot_folder_structure, display_folder_contents, plot_treemap
from sklearn.feature_extraction.text import TfidfVectorizer

def main():
    st.set_page_config(layout="wide", page_title="Bookmark Folder Structure Generator")
    st.title("Bookmark Folder Structure Generator")

    # Load data
    bookmarks_df, _ = load_preprocessed_data()
    if bookmarks_df is None:
        st.error("Failed to load preprocessed data. Please check the data file.")
        return

    # Display data in a collapsible section
    with st.expander("View Bookmarks Data", expanded=False):
        st.dataframe(bookmarks_df)

    # Embedding
    st.header("Embedding and Clustering")
    
    # Create tabs for different embedding methods
    tab1, tab2 = st.tabs(["TF-IDF", "Transformer"])
    
    with tab1:
        generate_embeddings_tab("TF-IDF", bookmarks_df)
    
    with tab2:
        generate_embeddings_tab("Transformer", bookmarks_df)
    
    # Clustering and Visualization (outside of tabs, will use the active embeddings)
    if 'active_embeddings' in st.session_state:
        clustering_and_visualization(bookmarks_df)

def generate_embeddings_tab(method, bookmarks_df):
    st.subheader(f"{method} Embedding")
    if st.button(f"Generate {method} Embeddings", key=f'generate_embeddings_{method}'):
        texts = (bookmarks_df['title'] + " " + bookmarks_df['url'] + " " + bookmarks_df['tags']).tolist()
        
        # Create a progress bar
        progress_bar = st.progress(0)
        
        # Generate embeddings with progress updates
        if method == "TF-IDF":
            vectorizer = TfidfVectorizer()
            embeddings = vectorizer.fit_transform(texts)
            embeddings = embeddings.toarray()  # Convert sparse matrix to dense array
        else:  # Transformer
            embeddings = []
            for i, text in enumerate(texts):
                embedding = generate_embeddings([text], method=method)[0]
                embeddings.append(embedding)
                progress_bar.progress((i + 1) / len(texts))
            embeddings = np.array(embeddings)
        
        st.success("Embeddings generated successfully!")
        save_preprocessed_data(bookmarks_df, embeddings)
        st.session_state['active_embeddings'] = embeddings
        st.session_state['active_method'] = method
        st.experimental_rerun()  # Rerun to update the clustering section

def clustering_and_visualization(bookmarks_df):
    st.header("Clustering and Visualization")
    
    clustering_method = st.selectbox("Clustering Method", list(CLUSTERING_METHODS.keys()), key='clustering_method')
    params = CLUSTERING_METHODS[clustering_method].get_params(key_prefix='clustering')
    
    if st.button("Generate Folder Structure", key='generate_structure'):
        with st.spinner("Generating folder structure..."):
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
        plot_folder_structure(st.session_state['hierarchy'], bookmarks_df)
        
        st.subheader("Treemap Visualization")
        plot_treemap(st.session_state['hierarchy'], bookmarks_df)
        
        st.subheader("Folder Contents")
        display_folder_contents(st.session_state['hierarchy'], bookmarks_df)

def create_folder_structure_dict(hierarchy, bookmarks_df):
    folder_structure = {}
    for cluster, indices in hierarchy.items():
        if cluster == -1:
            folder_name = "Uncategorized"
        else:
            folder_name = f"Folder {cluster}"
        
        folder_structure[folder_name] = [
            {
                "title": bookmarks_df.iloc[idx]['title'],
                "url": bookmarks_df.iloc[idx]['url'],
                "tags": bookmarks_df.iloc[idx]['tags']
            }
            for idx in indices
        ]
    return folder_structure

if __name__ == "__main__":
    main()