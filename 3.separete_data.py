import pandas as pd
from datetime import datetime, timedelta

file_path = "cleaned_posts.csv" 
df = pd.read_csv(file_path)

df = df.rename(columns={'id': 'id_text'})

df = df[df['text'].astype(str).str.len() >= 10]

df['date'] = pd.to_datetime(df['created_utc'], unit='s')

date_center = datetime(2024, 11, 5)
date_before_start = date_center - timedelta(days=100)  
date_before_end = date_center - timedelta(days=1)      

date_after_start = date_center + timedelta(days=1)  
date_after_end = date_center + timedelta(days=100)    

df_before = df[(df['date'] >= date_before_start) & (df['date'] <= date_before_end)]
df_after = df[(df['date'] >= date_after_start) & (df['date'] <= date_after_end)]

df_before.to_csv("cleaned_posts_before.csv", index=False)
df_after.to_csv("cleaned_posts_after.csv", index=False)


