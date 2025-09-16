import pandas as pd
import numpy as np
import re
import string
import time
from bs4 import BeautifulSoup
import spacy
nlp = spacy.load('en_core_web_sm')
from scipy.sparse import csr_matrix
import scipy.sparse as sps
import matplotlib.pyplot as plt
import leidenalg
import igraph as ig
import pickle
from IPython.display import HTML
import matplotlib as mpl
import matplotlib.cm
import umap
import seaborn as sns
from IPython.core.display import display
from IPython.core.display import display, HTML

from bertopic_try import BERTopic

class CleanText:
    def __init__(self, text, POS_KEEP = ["ADJ","ADV","NOUN","PROPN","VERB"]):
        tic = time.time()
        self.text = list(text)
        sup_clean = [self._superficial_cleaning(i) for i in self.text]
        self.text_clean = [self._deep_cleaning(i,POS_KEEP) for i in sup_clean]
        self.pos_clean = [self._deep_cleaning_pos(i,POS_KEEP) for i in sup_clean]
        print(f'Cleaning text: execution time {time.time()-tic} [s]')

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

    def _deep_cleaning(self, selftext,POS_KEEP):
        outtext = ' '.join([token.lemma_ for token in nlp(selftext) if token.pos_ in POS_KEEP])
        return outtext

    def _deep_cleaning_pos(self, selftext,POS_KEEP):
        outtext = [' '.join([token.lemma_,token.pos_]) for token in nlp(selftext) if token.pos_ in POS_KEEP]
        return outtext

def logg(x):
    y = np.log(x)
    y[x==0] = 0
    return y

def nmi_fn(A):
    aw = A.sum(axis=1).flatten()
    ac = A.sum(axis=0).flatten()
    Hc = np.multiply(ac,-logg(ac)).sum()
    A2 = ((A/ac).T/aw).T
    A2.data = logg(A2.data)
    y = (A.multiply(A2)).sum()/Hc
    return y

def modularity_fn(A):
    y = A.trace()-(A.sum(axis=0)*A.sum(axis=1)).item()
    return y

def ncut_fn(A):
    y = ((A.sum(axis=0)-A.diagonal())/A.sum(axis=0)).mean()
    return y

def pagerank_fn(M,q,c=.85,it=60):
    r = q.copy()
    for k in range(it):
      r = c*M.dot(r) + (1-c)*q
    return r

def _infomap_fn(v):
    y = -(v.data*logg(v.data/v.sum())).sum()
    return y

def infomap_rank_fn(Pdd):
    pd = Pdd.sum(axis=0).flatten()
    M = sps.csr_matrix(Pdd/pd)
    G = ig.Graph.Adjacency((M > 0).toarray().tolist())
    G.es['weight'] = np.array(M[M.nonzero()])[0]
    r = G.pagerank(weights='weight')
    r = (sps.csr_matrix(np.array(r))).T
    return r

def _hlstr(string: str , color: str = 'white'):
    return f"<span style=\"color:{color}\">{string} </span>"

def _colorize(attrs: np.ndarray, cmap: str = 'PiYG'):
    norm = mpl.colors.Normalize(vmin=-1, vmax=1)
    cmap = matplotlib.colormaps.get_cmap(cmap)
    return list(map(lambda x: mpl.colors.rgb2hex(cmap(norm(x))), attrs))

def _display_html(tokens, attrs, co=1):
    return HTML("".join(list(map(_hlstr, tokens, _colorize(co*attrs)))))
