import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import community as community_louvain

file_before = pd.read_csv('cleaned_posts_before.csv')
file_after = pd.read_csv('cleaned_posts_after.csv')

def draw_graph(data, title):
    G = nx.Graph()

    data = data.dropna(subset=['author', 'subreddit'])

    for _, row in data.iterrows():
        user = row['author']
        subreddit = row['subreddit']
        G.add_node(subreddit)
        G.add_node(user)
        G.add_edge(subreddit, user)

    partition = community_louvain.best_partition(G)

    plt.figure(figsize=(12, 12))

    pos = nx.spring_layout(G, k=0.3, iterations=20)  

    subreddit_nodes = [node for node in G.nodes() if node not in data['author'].unique()]

    colors = [partition.get(node, 0) for node in subreddit_nodes] 

    node_size = 100
    font_size = 10

    nx.draw_networkx_nodes(G, pos, nodelist=subreddit_nodes, node_size=node_size, node_color=colors, cmap=plt.cm.jet, alpha=0.7)
    
    nx.draw_networkx_labels(G, pos, labels={node: node for node in subreddit_nodes}, font_size=font_size, font_color='black')

    nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5, edge_color='gray')

    plt.title(title, fontsize=14)

    plt.show()


draw_graph(file_before, "Graph for Data Before November 6")

draw_graph(file_after, "Graph for Data After November 6")
