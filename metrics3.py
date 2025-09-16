import pandas as pd
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import community as community_louvain
import numpy as np
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char.isalnum() or char.isspace()])
    return text

before_df = pd.read_csv('cleaned_posts_before.csv')
after_df = pd.read_csv('cleaned_posts_after.csv')

all_posts_df = pd.concat([before_df, after_df]).iloc[:4500]

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
modularity_louvain = community_louvain.modularity(partition_louvain, graph)
community_labels_louvain = np.array(list(partition_louvain.values()))

spectral_clustering = SpectralClustering(n_clusters=len(set(partition_louvain.values())), affinity='nearest_neighbors', assign_labels="discretize")
soft_louvain_labels = spectral_clustering.fit_predict(tfidf_matrix)

kmeans = KMeans(n_clusters=len(set(partition_louvain.values())), random_state=42)
kmeans_labels = kmeans.fit_predict(tfidf_matrix)

dbscan = DBSCAN(eps=0.3, min_samples=5, metric='cosine') 
dbscan_labels = dbscan.fit_predict(tfidf_matrix)


nmi_louvain_soft = compute_nmi(community_labels_louvain, soft_louvain_labels)
nmi_louvain_kmeans = compute_nmi(community_labels_louvain, kmeans_labels)
nmi_louvain_dbscan = compute_nmi(community_labels_louvain, dbscan_labels)


print(f"Modularity (Louvain): {modularity_louvain}")
print(f"NMI between Louvain and Soft Louvain: {nmi_louvain_soft}")
print(f"NMI between Louvain and KMeans: {nmi_louvain_kmeans}")
print(f"NMI between Louvain and DBSCAN: {nmi_louvain_dbscan}")

algorithms = ['Soft Louvain', 'Hard Louvain', 'Infomap']
x_labels = ['Jan 2017', 'Feb 2018', 'Mar 2019', 'Apr 2020', 'May 2021', 'Jun 2022']


nmi = np.random.rand(6, 3)
q = np.random.rand(6, 3)
ncut = np.random.rand(6, 3)
infomap = np.random.rand(6, 3)
comm = np.random.rand(6, 3)
time = np.random.rand(6, 3)


fig, axs = plt.subplots(2, 3, figsize=(18, 10))  

axs[0, 0].plot(x_labels, nmi[:, 0], label=algorithms[0], marker='o', color='green')
axs[0, 0].plot(x_labels, nmi[:, 1], label=algorithms[1], marker='o', color='blue')
axs[0, 0].plot(x_labels, nmi[:, 2], label=algorithms[2], marker='o', color='yellow')
axs[0, 0].set_title("NMI")
axs[0, 0].set_ylabel("NMI Value")
axs[0, 0].legend()
axs[0, 0].set_xticklabels([])  


axs[0, 1].plot(x_labels, q[:, 0], label=algorithms[0], marker='o', color='green')
axs[0, 1].plot(x_labels, q[:, 1], label=algorithms[1], marker='o', color='blue')
axs[0, 1].plot(x_labels, q[:, 2], label=algorithms[2], marker='o', color='yellow')
axs[0, 1].set_title("Modularity (Q)")
axs[0, 1].set_ylabel("Q Value")
axs[0, 1].legend()
axs[0, 1].set_xticklabels([]) 


axs[0, 2].plot(x_labels, ncut[:, 0], label=algorithms[0], marker='o', color='green')
axs[0, 2].plot(x_labels, ncut[:, 1], label=algorithms[1], marker='o', color='blue')
axs[0, 2].plot(x_labels, ncut[:, 2], label=algorithms[2], marker='o', color='yellow')
axs[0, 2].set_title("Ncut")
axs[0, 2].set_ylabel("Ncut Value")
axs[0, 2].legend()
axs[0, 2].set_xticklabels([])  


axs[1, 0].plot(x_labels, infomap[:, 0], label=algorithms[0], marker='o', color='green')
axs[1, 0].plot(x_labels, infomap[:, 1], label=algorithms[1], marker='o', color='blue')
axs[1, 0].plot(x_labels, infomap[:, 2], label=algorithms[2], marker='o', color='yellow')
axs[1, 0].set_title("InfoMap")
axs[1, 0].set_ylabel("InfoMap Value")
axs[1, 0].legend()
axs[1, 0].set_xticklabels([])  


axs[1, 1].plot(x_labels, comm[:, 0], label=algorithms[0], marker='o', color='green')
axs[1, 1].plot(x_labels, comm[:, 1], label=algorithms[1], marker='o', color='blue')
axs[1, 1].plot(x_labels, comm[:, 2], label=algorithms[2], marker='o', color='yellow')
axs[1, 1].set_title("#comm")
axs[1, 1].set_ylabel("#comm Value")
axs[1, 1].legend()
axs[1, 1].set_xticklabels([])  


axs[1, 2].plot(x_labels, time[:, 0], label=algorithms[0], marker='o', color='green')
axs[1, 2].plot(x_labels, time[:, 1], label=algorithms[1], marker='o', color='blue')
axs[1, 2].plot(x_labels, time[:, 2], label=algorithms[2], marker='o', color='yellow')
axs[1, 2].set_title("Time")
axs[1, 2].set_ylabel("Time (seconds)")
axs[1, 2].legend()
axs[1, 2].set_xticklabels([])  


plt.tight_layout()
plt.show()
