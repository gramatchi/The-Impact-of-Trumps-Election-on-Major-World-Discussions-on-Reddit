import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import community as community_louvain
import umap
from node2vec import Node2Vec

file_before = pd.read_csv('cleaned_posts_before.csv')
file_after = pd.read_csv('cleaned_posts_after.csv')

def draw_graph_with_umap(data, title):
    G = nx.Graph()

    data = data.dropna(subset=['author', 'subreddit'])

    for _, row in data.iterrows():
        user = row['author']
        subreddit = row['subreddit']
        G.add_edge(subreddit, user)  

    partition = community_louvain.best_partition(G)

    node2vec = Node2Vec(G, dimensions=16, walk_length=30, num_walks=200, workers=4)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)

    embeddings = [model.wv[node] for node in G.nodes()]

    umap_layout = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(embeddings)

    pos = {node: umap_layout[i] for i, node in enumerate(G.nodes())}

    subreddit_nodes = [node for node in G.nodes() if node in data['subreddit'].unique()]

    colors = [partition.get(node, 0) for node in subreddit_nodes]

    plt.figure(figsize=(12, 12))

    nx.draw_networkx_nodes(G, pos, nodelist=subreddit_nodes, node_size=100, node_color=colors, cmap=plt.cm.jet, alpha=0.7)
    
    nx.draw_networkx_labels(G, pos, labels={node: node for node in subreddit_nodes}, font_size=10, font_color='black')

    nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5, edge_color='gray')

    plt.title(title, fontsize=14)
    plt.show()

draw_graph_with_umap(file_before, "Graph with UMAP - Before November 6")
draw_graph_with_umap(file_after, "Graph with UMAP - After November 6")
