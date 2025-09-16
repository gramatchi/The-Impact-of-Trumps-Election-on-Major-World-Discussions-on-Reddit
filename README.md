# The Impact of Trump's Election on Major World Discussions on Reddit

## 📌 Project Description
This repository contains a **student project** in the field of **Network Science**.  
The project analyzes how popular topics on Reddit changed **100 days before** and **100 days after Donald Trump’s election**.  

The main goal was to understand how political events influenced:
- the **topics of discussions**,  
- the **structure of user interactions**,  
- the **emotional tone** of conversations.  

A detailed presentation and results are provided in the PDF file.  
The dataset (`.csv` and `.xlsx`) includes Reddit posts and extracted topics used in the analysis.

## ⚙️ Methods
The following methods and tools were applied:

- **Data collection from Reddit** via API (`PRAW`)  
- **Text preprocessing:** removing HTML, URLs, special characters, lemmatization  
- **Topic modeling:**  
  - Louvain clustering for community detection  
  - BERTopic for thematic clustering and topic dynamics  
- **Network analysis:** user–subreddit interaction graphs with `NetworkX` and `UMAP`  
- **Sentiment and emotional analysis:**  
  - LIWC for linguistic evaluation  
  - BERT for sentiment classification  
- **Visualization:** word clouds, topic clusters, interaction graphs  

## 📂 Repository Structure
### Data Files
- `cleaned_posts_before.csv` – processed posts (100 days before election)  
- `cleaned_posts_after.csv` – processed posts (100 days after election)  
- `emotional_changes_analysis.csv` – emotional and sentiment changes per subreddit  
- `.xlsx` files – alternative structured datasets for analysis  

### Python Scripts
1. **`scratch_data.py`** – collects raw Reddit data via API  
2. **`clean_data.py`** – cleans the dataset (HTML, URLs, deduplication, lemmatization)  
3. **`separate_data.py`** – splits the dataset into *before* and *after* election periods  
4. **`topic_detection.py`** – extracts and clusters topics using BERTopic and Louvain  
5. **`bert-agent.py`** – applies BERT sentiment analysis  
6. **`bipartite.py`** – builds bipartite graphs of *subreddits vs. posts*  
7. **`liwc_try.py`** – combines LIWC analysis and BERT sentiment analysis  
8. **`metrics.py`** – evaluates clustering quality  
9. **`umap-try.py`** – visualizes subreddit clusters with UMAP dimensionality reduction  
10. **`visualize_emotions.py`** – plots emotional changes (posemo, negemo, anger, anxiety, swear) per subreddit  
11. **`word_cloud.py`** – generates word clouds from text data (*before/after election*)  

## ⚠️ Limitations
- Short observation period (100 days)  
- Reddit is not fully representative of global discussions  
- Other external events may also have influenced the results  

## 👥 Authors
- Nichita Gramatchi  
- Malikov Andrey  
