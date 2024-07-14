import streamlit as st
import plotly.graph_objs as go
import networkx as nx
import plotly.express as px
from collections import Counter

def create_folder_structure(hierarchy, bookmarks_df, depth_limit=None):
    G = nx.Graph()
    G.add_node("Root")
    
    def add_nodes(node, parent="Root", current_depth=0):
        if depth_limit is not None and current_depth > depth_limit:
            return
        
        if 'children' not in node:
            # This is a leaf node (bookmark)
            bookmark = bookmarks_df.iloc[node['node_id']]
            G.add_node(node['node_id'], title=bookmark['title'], type='bookmark')
            G.add_edge(parent, node['node_id'])
        else:
            # This is an internal node (folder)
            folder_name = get_folder_name(node, bookmarks_df)
            G.add_node(folder_name, type='folder')
            G.add_edge(parent, folder_name)
            for child in node['children']:
                add_nodes(child, folder_name, current_depth + 1)
    
    add_nodes(hierarchy)
    return G

def get_folder_name(node, bookmarks_df):
    if 'children' not in node:
        return bookmarks_df.iloc[node['node_id']]['title']
    
    # Get all words from titles and tags of bookmarks in this folder
    words = []
    def collect_words(n):
        if 'children' not in n:
            bookmark = bookmarks_df.iloc[n['node_id']]
            words.extend(bookmark['title'].split())
            words.extend(bookmark['tags'].split())
        else:
            for child in n['children']:
                collect_words(child)
    
    collect_words(node)
    
    # Count words and get the most common ones
    word_counts = Counter(words)
    common_words = [word for word, count in word_counts.most_common(3) if count > 1]
    
    if common_words:
        return " ".join(common_words)
    else:
        return f"Folder {node['node_id']}"

def plot_folder_structure(hierarchy, bookmarks_df, depth_limit=None):
    G = create_folder_structure(hierarchy, bookmarks_df, depth_limit)
    
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
    node_text = []
    node_hovertext = []
    node_size = []
    node_color = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        node_type = G.nodes[node].get('type', '')
        if node == "Root":
            node_text.append("Root")
            node_hovertext.append("Root")
            node_size.append(30)
            node_color.append('#1F77B4')
        elif node_type == 'folder':
            node_text.append(node)
            node_hovertext.append(node)
            node_size.append(20)
            node_color.append('#FF9900')
        else:  # Bookmark
            node_text.append('')  # Empty text for bookmarks
            node_hovertext.append(G.nodes[node]['title'])
            node_size.append(10)
            node_color.append('#2CA02C')
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        hovertext=node_hovertext,
        textposition="top center",
        marker=dict(
            showscale=False,
            size=node_size,
            color=node_color,
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

def display_folder_contents(hierarchy, bookmarks_df, depth=0):
    if 'children' not in hierarchy:
        bookmark = bookmarks_df.iloc[hierarchy['node_id']]
        st.write(f"{'  ' * depth}- [{bookmark['title']}]({bookmark['url']})")
    else:
        folder_name = get_folder_name(hierarchy, bookmarks_df)
        st.write(f"{'  ' * depth}- {folder_name} ({hierarchy['size']} bookmarks)")
        for child in hierarchy['children']:
            display_folder_contents(child, bookmarks_df, depth + 1)

def plot_treemap(hierarchy, bookmarks_df):
    def build_treemap_data(node, parent=""):
        if 'children' not in node:
            bookmark = bookmarks_df.iloc[node['node_id']]
            return [{
                "ids": bookmark['title'],
                "labels": bookmark['title'],
                "parents": parent,
                "values": 1
            }]
        else:
            folder_name = get_folder_name(node, bookmarks_df)
            data = [{
                "ids": folder_name,
                "labels": folder_name,
                "parents": parent,
                "values": node['size']
            }]
            for child in node['children']:
                data.extend(build_treemap_data(child, folder_name))
            return data

    treemap_data = build_treemap_data(hierarchy)
    
    fig = go.Figure(go.Treemap(
        ids=[item["ids"] for item in treemap_data],
        labels=[item["labels"] for item in treemap_data],
        parents=[item["parents"] for item in treemap_data],
        values=[item["values"] for item in treemap_data],
    ))
    
    fig.update_layout(
        title='Bookmark Folder Treemap',
        margin = dict(t=50, l=25, r=25, b=25)
    )
    
    st.plotly_chart(fig, use_container_width=True)