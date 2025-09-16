import time
import pandas as pd
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import community as community_louvain
from sklearn.metrics import normalized_mutual_info_score
from sklearn.cluster import SpectralClustering, KMeans, DBSCAN
import hdbscan
import matplotlib.pyplot as plt
import numpy as np

algorithms = ['Louvain', 'Spectral', 'KMeans', 'DBSCAN', 'HDBSCAN']

def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char.isalnum() or char.isspace()])
    return text

before_df = pd.read_csv('cleaned_posts_before.csv')
after_df = pd.read_csv('cleaned_posts_after.csv')
all_posts_df = pd.concat([before_df, after_df]).iloc[:4500]

all_posts_df['cleaned_text'] = all_posts_df['text'].dropna().apply(preprocess_text)
all_posts_df['text_length'] = all_posts_df['cleaned_text'].apply(len)
all_posts_df['cleaned_text'] = all_posts_df['cleaned_text'].apply(lambda x: x[:1000])

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(all_posts_df['cleaned_text'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

threshold = 0.05
graph = nx.Graph()
for idx, row in all_posts_df.iterrows():
    graph.add_node(idx, text=row['text'])
for i in range(len(cosine_sim)):
    for j in range(i + 1, len(cosine_sim)):
        if cosine_sim[i, j] > threshold:
            graph.add_edge(i, j, weight=cosine_sim[i, j])

def compute_nmi_safe(true_labels, pred_labels):
    min_len = min(len(true_labels), len(pred_labels))
    true_labels, pred_labels = true_labels[:min_len], pred_labels[:min_len]
    return normalized_mutual_info_score(true_labels, pred_labels) if len(set(pred_labels)) > 1 else 0.0

start_time = time.time()
partition_louvain = community_louvain.best_partition(graph)
modularity_louvain = community_louvain.modularity(partition_louvain, graph)
community_labels_louvain = np.array(list(partition_louvain.values()))
time_louvain = time.time() - start_time

start_time = time.time()
n_clusters = len(set(community_labels_louvain))
spectral_clustering = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors')
spectral_labels = spectral_clustering.fit_predict(tfidf_matrix)
time_spectral = time.time() - start_time

start_time = time.time()
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans_labels = kmeans.fit_predict(tfidf_matrix)
time_kmeans = time.time() - start_time

start_time = time.time()
dbscan = DBSCAN(eps=0.2, min_samples=3, metric='cosine')
dbscan_labels = dbscan.fit_predict(tfidf_matrix)
time_dbscan = time.time() - start_time

start_time = time.time()
hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=5, metric='euclidean')
hdbscan_labels = hdbscan_clusterer.fit_predict(tfidf_matrix.toarray())
time_hdbscan = time.time() - start_time

nmi_values = [
    compute_nmi_safe(community_labels_louvain, spectral_labels),
    compute_nmi_safe(community_labels_louvain, kmeans_labels),
    compute_nmi_safe(community_labels_louvain, dbscan_labels),
    compute_nmi_safe(community_labels_louvain, hdbscan_labels)
]

modularity_values = [
    modularity_louvain,
    community_louvain.modularity({i: lbl for i, lbl in enumerate(spectral_labels)}, graph) if len(set(spectral_labels)) > 1 else 0.0,
    community_louvain.modularity({i: lbl for i, lbl in enumerate(kmeans_labels)}, graph) if len(set(kmeans_labels)) > 1 else 0.0,
    0.0,
    0.0
]

num_communities = [
    len(set(community_labels_louvain)),
    len(set(spectral_labels)),
    len(set(kmeans_labels)),
    len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0),
    len(set(hdbscan_labels)) - (1 if -1 in hdbscan_labels else 0)
]

def normalize(data):
    min_val, max_val = min(data), max(data)
    return [(x - min_val) / (max_val - min_val) if max_val > min_val else 0 for x in data]

nmi_values = normalize(nmi_values)
modularity_values = normalize(modularity_values)
time_values = np.log1p([time_louvain, time_spectral, time_kmeans, time_dbscan, time_hdbscan])
num_communities = normalize(num_communities)
values = [nmi_values, modularity_values, num_communities, time_values]

for i in range(len(values)):
    if len(values[i]) < len(algorithms):
        values[i] = [0.0] * (len(algorithms) - len(values[i])) + values[i]

fig, axs = plt.subplots(1, 4, figsize=(20, 5))
color_map = {'Louvain': 'black', 'Spectral': 'blue', 'KMeans': 'green', 'DBSCAN': 'red', 'HDBSCAN': 'purple'}
colors = [color_map[alg] for alg in algorithms]

titles = ["NMI Comparison", "Modularity (Q)", "Number of Communities", "Execution Time"]

for i, ax in enumerate(axs):
    ax.bar(algorithms, values[i], color=colors)
    ax.set_title(titles[i])
    ax.set_ylabel("Value")
    if len(values[i]) > 0:
        ax.set_ylim(min(values[i]) * 0.9, max(values[i]) * 1.1)
    for bar in ax.patches:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,  
                f'{bar.get_height():.2f}', ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()