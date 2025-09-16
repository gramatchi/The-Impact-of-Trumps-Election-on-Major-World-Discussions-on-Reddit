import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

def create_and_compare_graphs(file_path, title):
    df = pd.read_csv(file_path)

    G = nx.Graph()

    subreddits = df['subreddit'].value_counts().index[:30] 
    filtered_df = df[df['subreddit'].isin(subreddits)]

    filtered_df = filtered_df.sort_values(by='num_comments', ascending=False).head(300)
    post_ids = filtered_df['id_text'].unique()

    G.add_nodes_from(subreddits, bipartite=0) 
    G.add_nodes_from(post_ids, bipartite=1)  

    edges = [(row['subreddit'], row['id_text']) for _, row in filtered_df.iterrows()]
    G.add_edges_from(edges)


    pos_kamada = nx.kamada_kawai_layout(G)  
    pos_spring = nx.spring_layout(G, k=0.3, seed=42)  

    fig, axs = plt.subplots(1, 2, figsize=(18, 9))
    layouts = {'Kamada-Kawai Layout': pos_kamada, 'Spring Layout': pos_spring}

    for ax, (layout_name, pos) in zip(axs, layouts.items()):
        ax.set_title(f"{title} - {layout_name}", fontsize=14, fontweight='bold')
        
        node_colors = ["red" if node in subreddits else "blue" for node in G.nodes]
        node_sizes = [300 if node in subreddits else 50 for node in G.nodes]

        nx.draw(G, pos, ax=ax, with_labels=False, node_size=node_sizes, 
                node_color=node_colors, edge_color="gray", alpha=0.3, width=0.5)

        subreddit_labels = {n: n for n in subreddits}
        nx.draw_networkx_labels(G, pos, ax=ax, labels=subreddit_labels, font_size=10, font_color='red', font_weight='bold')

    plt.tight_layout()
    plt.show()

file_before = "cleaned_posts_before.csv"
file_after = "cleaned_posts_after.csv"

create_and_compare_graphs(file_after, "Bipartite Graph: Subreddits - Posts (After)")

create_and_compare_graphs(file_before, "Bipartite Graph: Subreddits - Posts (Before)")
