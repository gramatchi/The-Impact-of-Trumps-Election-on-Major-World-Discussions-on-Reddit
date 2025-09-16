import pandas as pd
from transformers import pipeline

data_path = 'cleaned_posts_after.csv'
df = pd.read_csv(data_path)

print(df.head())

agent = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

if 'text' in df.columns: 
    df['bert_analysis'] = df['text'].apply(lambda x: agent(x)[0]['label'])
    print(df[['text', 'bert_analysis']])  
else:
    print("no column text")
