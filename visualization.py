import streamlit as st
import plotly.express as px
import plotly.graph_objs as go
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples
import networkx as nx
from scipy.sparse import csr_matrix

def visualize_clusters(embeddings, labels, bookmarks_df):
    # Reduce dimensionality for visualization
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    
    # Create a DataFrame for plotting
    plot_df = bookmarks_df.copy()
    plot_df['x'] = embeddings_2d[:, 0]
    plot_df['y'] = embeddings_2d[:, 1]
    plot_df['cluster'] = labels
    
    # Create the scatter plot
    fig = px.scatter(plot_df, x='x', y='y', color='cluster', hover_data=['title', 'url'])
    fig.update_layout(title="Cluster Visualization")
    st.plotly_chart(fig, use_container_width=True)

def plot_silhouette_scores(embeddings, labels):
    silhouette_values = silhouette_samples(embeddings, labels)
    
    y_lower, y_upper = 0, 0
    yticks = []
    cluster_silhouette_values = []
    cluster_labels = []
    
    for cluster in sorted(set(labels)):
        cluster_values = silhouette_values[labels == cluster]
        cluster_values.sort()
        
        y_upper += len(cluster_values)
        cluster_silhouette_values.extend(cluster_values)
        cluster_labels.extend([cluster] * len(cluster_values))
        
        yticks.append((y_lower + y_upper) / 2)
        y_lower = y_upper + 10
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=cluster_silhouette_values,
        y=list(range(len(silhouette_values))),
        orientation='h',
        marker=dict(color=cluster_labels, colorscale='Viridis'),
        showlegend=False
    ))
    
    fig.update_layout(
        title="Silhouette Plot",
        xaxis_title="Silhouette coefficient values",
        yaxis_title="Cluster label",
        yaxis=dict(tickmode='array', tickvals=yticks, ticktext=sorted(set(labels))),
        height=600,
        width=800
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_minimum_spanning_tree(mst):
    # Convert the MST to a NetworkX graph
    G = nx.from_scipy_sparse_array(csr_matrix(mst))
    
    # Calculate layout
    pos = nx.spring_layout(G)
    
    # Create edges trace
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')
    
    # Create nodes trace
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            )
        )
    )
    
    # Color nodes by number of connections
    node_adjacencies = []
    for node, adjacencies in G.adjacency():
        node_adjacencies.append(len(adjacencies))
    
    node_trace.marker.color = node_adjacencies
    
    # Create the figure
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Nearest Neighbors Graph',
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    
    st.plotly_chart(fig, use_container_width=True)