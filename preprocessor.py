import pandas as pd 
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from tensorflow.keras.preprocessing.text import Tokenizer
import streamlit as st

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
url_pattern = re.compile(re.compile(r'https?://\S+|www\.S+'))

def clean_data(df):
    doc = []
    st.write(df.columns)
    for entry in df['text']:
        
        tokens = entry.split()
        tokens = [word for word in tokens if word.isalpha()]
        tokens = [lemmatizer.lemmatize(w.lower()) for w in tokens]
        tokens = [w for w in tokens if not w in stop_words]
        tokens = [word.lower() for word in tokens if len(word) > 1]
        tokens = [url_pattern.sub('', w) for w in tokens]
        doc.append(' '.join(tokens))                         
    df["text"]= doc 
    return df

############################################
# ENCODER SECTION #
def encoder(df,mode='binary'):
    t = Tokenizer()
    entries = [entry for entry in df[feature]]
    t.fit_on_texts(entries)
    feature_names = list(t.word_index.keys())
    encoded = t.texts_to_matrix(entries, mode=mode)[:, 1:]
    return encoded, feature_names



