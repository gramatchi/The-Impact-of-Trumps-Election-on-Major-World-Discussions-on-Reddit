import pandas as pd
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import community as community_louvain
from sklearn.metrics import normalized_mutual_info_score
from sklearn.cluster import SpectralClustering, KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN

def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char.isalnum() or char.isspace()])
    return text


before_df = pd.read_csv('cleaned_posts_before.csv')
after_df = pd.read_csv('cleaned_posts_after.csv')

all_posts_df = pd.concat([before_df, after_df])

all_posts_df = all_posts_df.iloc[:4500]

all_posts_df['cleaned_text'] = all_posts_df['text'].apply(preprocess_text)

max_post_length = 1000
all_posts_df['text_length'] = all_posts_df['cleaned_text'].apply(len)
all_posts_df['cleaned_text'] = all_posts_df['cleaned_text'].apply(lambda x: x[:max_post_length])

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(all_posts_df['cleaned_text'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

threshold = 0.05 
graph = nx.Graph()

for idx, row in all_posts_df.iterrows():
    graph.add_node(idx, text=row['text'], keyword=row['keyword'])

for i in range(len(cosine_sim)):
    for j in range(i + 1, len(cosine_sim)):
        if cosine_sim[i, j] > threshold:
            graph.add_edge(i, j, weight=cosine_sim[i, j])

def compute_nmi(true_labels, pred_labels):
    min_len = min(len(true_labels), len(pred_labels))
    true_labels = true_labels[:min_len]
    pred_labels = pred_labels[:min_len]
    return normalized_mutual_info_score(true_labels, pred_labels)


partition_louvain = community_louvain.best_partition(graph)
modularity = community_louvain.modularity(partition_louvain, graph)
community_labels_louvain = np.array(list(partition_louvain.values()))


n_clusters = len(set(partition_louvain.values()))
spectral_clustering = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors')
ncut_spectral = spectral_clustering.fit_predict(tfidf_matrix)

kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans_labels = kmeans.fit_predict(tfidf_matrix)

dbscan = DBSCAN(eps=0.3, min_samples=5, metric='cosine')  
dbscan_labels = dbscan.fit_predict(tfidf_matrix)

nmi_louvain_spectral = compute_nmi(community_labels_louvain, ncut_spectral)
nmi_louvain_kmeans = compute_nmi(community_labels_louvain, kmeans_labels)
nmi_louvain_dbscan = compute_nmi(community_labels_louvain, dbscan_labels)


print(f"Modularity (Louvain): {modularity}")
print(f"NMI between Louvain and Spectral Clustering: {nmi_louvain_spectral}")
print(f"NMI between Louvain and KMeans: {nmi_louvain_kmeans}")
print(f"NMI between Louvain and DBSCAN: {nmi_louvain_dbscan}")
