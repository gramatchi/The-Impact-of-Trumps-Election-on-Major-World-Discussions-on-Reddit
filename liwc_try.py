import pandas as pd
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from transformers import pipeline
import liwc  
import matplotlib.pyplot as plt

nltk.download('punkt')

file_before = "cleaned_posts_before.csv"
file_after = "cleaned_posts_after.csv"

df_before = pd.read_csv(file_before)
df_after = pd.read_csv(file_after)

required_columns = {'subreddit', 'text'}
assert required_columns.issubset(df_before.columns), "The 'before' file does not contain the required columns!"
assert required_columns.issubset(df_after.columns), "The 'after' file does not contain the required columns!"

liwc_dict, _ = liwc.read_dic("liwc_dict.dic")  

def liwc_analysis(text):
    words = word_tokenize(text.lower())  
    emotions = {'posemo': 0, 'negemo': 0, 'anger': 0, 'anx': 0, 'swear': 0, 'agency': 0}
    
    for word in words:
        if word in liwc_dict:
            for category in liwc_dict[word]:
                if category in emotions:
                    emotions[category] += 1
    return emotions

sentiment_pipeline = pipeline("sentiment-analysis")

def bert_sentiment(text):
    result = sentiment_pipeline(text[:512])  
    return result[0]['label'], result[0]['score']

def analyze_dataframe(df):
    results = []
    for _, row in df.iterrows():
        subreddit = row['subreddit']
        text = row['text']
        
        liwc_scores = liwc_analysis(text)
        
        sentiment_label, sentiment_score = bert_sentiment(text)

        results.append({
            'subreddit': subreddit,
            'posemo': liwc_scores['posemo'],
            'negemo': liwc_scores['negemo'],
            'anger': liwc_scores['anger'],
            'anx': liwc_scores['anx'],
            'swear': liwc_scores['swear'],
            'agency': liwc_scores['agency'],
            'bert_sentiment': sentiment_label,
            'bert_confidence': sentiment_score
        })
    return pd.DataFrame(results)

df_before_analysis = analyze_dataframe(df_before)
df_after_analysis = analyze_dataframe(df_after)

numeric_cols = ['posemo', 'negemo', 'anger', 'anx', 'swear', 'agency']

df_merged = df_before_analysis.groupby('subreddit')[numeric_cols].mean().merge(
    df_after_analysis.groupby('subreddit')[numeric_cols].mean(),
    on='subreddit',
    suffixes=('_before', '_after')
)

for col in ['posemo', 'negemo', 'anger', 'anx', 'swear', 'agency']:
    df_merged[f'{col}_change'] = df_merged[f'{col}_after'] - df_merged[f'{col}_before']


print(df_merged.columns)

if 'bert_confidence_after' in df_merged.columns and 'bert_confidence_before' in df_merged.columns:
    df_merged['bert_sentiment_change'] = df_merged['bert_confidence_after'] - df_merged['bert_confidence_before']
else:
    print("Columns 'bert_confidence_after' and 'bert_confidence_before' are missing!")


df_merged.to_csv("emotional_changes_analysis.csv", index=True)

fig, axes = plt.subplots(3, 2, figsize=(12, 10))
axes = axes.flatten()

metrics = ['posemo', 'negemo', 'anger', 'anx', 'swear', 'agency']

for i, col in enumerate(metrics):
    df_merged[f'{col}_change'].plot(kind='bar', ax=axes[i], title=f"Change in {col.capitalize()} Score")

    axes[i].set_xlabel("Subreddit")
    axes[i].set_ylabel("Change Score")

plt.tight_layout()
plt.show()
