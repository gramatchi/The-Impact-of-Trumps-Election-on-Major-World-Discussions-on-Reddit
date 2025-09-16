import re
import time
import spacy
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from tqdm import tqdm

nlp = spacy.load("en_core_web_sm")

class CleanText:
    def __init__(self, text, POS_KEEP=["ADJ", "ADV", "NOUN", "PROPN", "VERB"]):
        tic = time.time()
        self.text = [i if isinstance(i, str) else "" for i in text]  
        sup_clean = [self._superficial_cleaning(i) for i in self.text]
        self.text_clean = [self._deep_cleaning(i, POS_KEEP) for i in sup_clean]


    def _superficial_cleaning(self, selftext):
        soup = BeautifulSoup(selftext, "html.parser")
        outtext = soup.get_text(separator=" ")
        outtext = re.sub(r'\[.*?\]', '', outtext)
        outtext = re.sub(r'http\S+', '', outtext)
        outtext = re.sub(r'www.\S+', '', outtext)
        outtext = outtext.replace('. com', '.com')
        outtext = re.sub(r'&amp;#x200B;\n\\', ' ', outtext)
        outtext = re.sub(r'‚Äú', ' ', outtext)
        outtext = re.sub(r'‚Äô', "’", outtext)
        outtext = re.sub(r' +', ' ', outtext)
        outtext = re.sub(r'\s{2,}', ' ', outtext)
        outtext = re.sub(r'&gt;', ' ', outtext)
        outtext = outtext.replace('-', ' ')
        outtext = outtext.replace('\\n', ' ').replace('\n', ' ').replace('\t',' ').replace('\\', ' ')
        Pattern_alpha = re.compile(r"([A-Za-z])\1{1,}", re.DOTALL)
        Pattern_Punct = re.compile(r'([.,/#!$%^&*?;:{}=_`~()+-])\1{1,}')
        outtext = Pattern_alpha.sub(r"\1\1", outtext)
        outtext = Pattern_Punct.sub(r'\1', outtext)
        outtext = re.sub(' {2,}',' ', outtext)
        pattern = re.compile(r'\s+')
        Without_whitespace = re.sub(pattern, ' ', outtext)
        outtext = Without_whitespace.replace('?', ' ? ').replace(')', ') ')
        return outtext

    def _deep_cleaning(self, selftext, POS_KEEP):
        outtext = ' '.join([token.lemma_ for token in nlp(selftext) if token.pos_ in POS_KEEP])
        return outtext

df = pd.read_csv('posts.xlsx')

df['text'] = df[['title', 'selftext']].fillna('').agg(' '.join, axis=1)

df['text'] = [CleanText([x]).text_clean[0] for x in tqdm(df['text'], desc='Cleaning Text')]

df.drop(columns=['title', 'selftext'], inplace=True)

df.to_csv('cleaned_posts.csv', index=False)

