import streamlit as st
import plotly.graph_objs as go
import networkx as nx

def create_folder_structure(hierarchy, bookmarks_df):
    G = nx.Graph()
    G.add_node("Root")
    
    for cluster, indices in hierarchy.items():
        if cluster == -1:
            folder_name = "Uncategorized"
        else:
            folder_name = f"Folder {cluster}"
        G.add_node(folder_name)
        G.add_edge("Root", folder_name)
        
        for idx in indices:
            bookmark = bookmarks_df.iloc[idx]
            G.add_node(bookmark['title'])
            G.add_edge(folder_name, bookmark['title'])
    
    return G

def plot_folder_structure(hierarchy, bookmarks_df):
    G = create_folder_structure(hierarchy, bookmarks_df)
    
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    
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
    
    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=[node if node.startswith("Folder") or node == "Root" or node == "Uncategorized" else "" for node in G.nodes()],
        textposition="top center",
        marker=dict(
            showscale=False,
            colorscale='YlGnBu',
            size=[20 if node.startswith("Folder") or node == "Root" or node == "Uncategorized" else 10 for node in G.nodes()],
            color=['#FF9900' if node.startswith("Folder") else '#1F77B4' if node == "Root" else '#D3D3D3' if node == "Uncategorized" else '#2CA02C' for node in G.nodes()],
            line_width=2
        )
    )
    
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Bookmark Folder Structure',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        annotations=[dict(
                            text="",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002)],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_folder_contents(hierarchy, bookmarks_df):
    st.write("Folder Structure:")
    for cluster, indices in hierarchy.items():
        if cluster == -1:
            folder_name = "Uncategorized"
        else:
            folder_name = f"Folder {cluster}"
        with st.expander(f"{folder_name} ({len(indices)} bookmarks)"):
            for idx in indices:
                bookmark = bookmarks_df.iloc[idx]
                st.write(f"- [{bookmark['title']}]({bookmark['url']})")