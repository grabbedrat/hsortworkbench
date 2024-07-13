import pandas as pd
import numpy as np
import pickle
import streamlit as st

def load_preprocessed_data():
    try:
        with open('preprocessed_bookmarks.pkl', 'rb') as f:
            data = pickle.load(f)
        
        # Check if data is a tuple (old format) or dict (new format)
        if isinstance(data, tuple):
            bookmarks_df, preprocessing_params = data
            embeddings = None  # Embeddings not available in this format
        elif isinstance(data, dict):
            bookmarks_df = data.get('bookmarks_df')
            embeddings = data.get('embeddings')
        else:
            raise ValueError("Unrecognized data format")
        
        if embeddings is None:
            st.warning("Embeddings not found in the preprocessed data. You may need to generate them.")
        
        return bookmarks_df, embeddings
    except FileNotFoundError:
        st.error("Preprocessed data file not found. Please make sure 'preprocessed_bookmarks.pkl' exists.")
        return None, None
    except Exception as e:
        st.error(f"Error loading preprocessed data: {str(e)}")
        return None, None

def save_preprocessed_data(bookmarks_df, embeddings=None):
    data = {
        'bookmarks_df': bookmarks_df,
        'embeddings': embeddings
    }
    with open('preprocessed_bookmarks.pkl', 'wb') as f:
        pickle.dump(data, f)
    st.success("Preprocessed data saved successfully.")