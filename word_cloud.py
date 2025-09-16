import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

file_path = 'cleaned_posts_after.csv' 
data = pd.read_csv(file_path)

text_data = " ".join(data['text'].dropna())

wordcloud = WordCloud(width=1600, height=800, background_color='white').generate(text_data)

plt.figure(figsize=(16, 8), facecolor=None)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()
