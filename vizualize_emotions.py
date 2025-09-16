import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("emotional_changes_analysis.csv")

fig, axes = plt.subplots(3, 2, figsize=(15, 10))

axes = axes.flatten()

metrics = ['posemo_change', 'negemo_change', 'anger_change', 'anx_change', 'swear_change']


for i, col in enumerate(metrics):
    df.sort_values(by=col, ascending=False, inplace=True)
    axes[i].barh(df['subreddit'], df[col], color='skyblue')
    axes[i].set_title(f"Change in {col.replace('_', ' ').capitalize()}")
    axes[i].set_xlabel("Change Score")
    axes[i].set_ylabel("Subreddit")

plt.tight_layout()
plt.show()
