import os
import pickle
from typing import List
from math import log
from collections import defaultdict

import nltk
import fugashi
import numpy as np
from tqdm import tqdm
from nltk.stem import porter, WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from . import file_utils

def normalizer(lang: str = "en"):
    if lang == "en":
        stemmer = porter.PorterStemmer()
        lemmatizer = WordNetLemmatizer()
    elif lang == "ja":
        tagger = fugashi.Tagger()
    else:
        raise NotImplementedError

    def fn(doc):
        if lang == "en":
            words = [w for w in nltk.wordpunct_tokenize(doc) if w.isalpha() and len(w) > 2]
            words = [lemmatizer.lemmatize(w, "v") for w in words]
            words = [lemmatizer.lemmatize(w, "n") for w in words]
            words = [stemmer.stem(w) for w in words]
        elif lang == "ja":
            words = [str(w.feature.lemma) for w in tagger(doc)]
            words = [w for w in words if w.isalpha() and w != "None"]
        return " ".join(words).lower()
        
    return fn

def normalize_text(doc_content_list: List[str], lang: str = "en", cache_dir="./tmp"):
    transformer = normalizer(lang=lang)
    res = [transformer(doc) for doc in tqdm(doc_content_list, desc="Normalizing")]
    file_utils.dump(res, os.path.join(cache_dir, f"normalized_text_{lang}.pck"))
    return res

def get_tfidf_score(texts, lang: str = "en", cache_dir="./tmp"):
    stop_words = None if lang != "en" else "english"
    vectorizer = TfidfVectorizer(tokenizer=lambda x : x.split(), stop_words=stop_words, min_df=10, max_df=0.7)
    X = vectorizer.fit_transform(texts)
    tf_vocab = vectorizer.vocabulary_
    print("Vocabulary size:", len(tf_vocab))
    with open(os.path.join(cache_dir, f"vocab_{lang}.txt"), "w") as f:
        f.write(" ".join(sorted(tf_vocab.keys())))

    return X, tf_vocab

def get_pmi(doc_content_list: List[str], vocab: List[str], window_size: int = 20):
    vocab = {w : i for i, w in enumerate(vocab)}
    doc_word_list = [[w for w in x.split() if w in vocab] for x in doc_content_list]
    windows = []
    for words in doc_word_list:
        length = len(words)
        if length <= window_size:
            windows.append(words)
        else:
            for j in range(length - window_size + 1):
                window = words[j: j + window_size]
                windows.append(window)

    word_window_freq = defaultdict(lambda : 0)
    for window in tqdm(windows, desc="Word frequency (windows)"):
        for w in set(window):
            word_window_freq[w] += 1

    word_pair_count = defaultdict(lambda : 0)
    for window in tqdm(windows, desc="Word pairs frequency (windows)"):
        for i in range(1, len(window)):
            for j in range(i):
                wi, wj = window[i], window[j]
                if wi == wj:
                    continue
                word_pair_count[(wi, wj)] += 1
                word_pair_count[(wj, wi)] += 1

    # pmi as weights
    num_window = len(windows)
    pmi_word_word = np.zeros((len(vocab), len(vocab)))
    for (wi, wj), count in word_pair_count.items():
        word_freq_i = word_window_freq[wi]
        word_freq_j = word_window_freq[wj]
        # pmi = log[p(i,j) / p(i)p(j)] = log[n * n(i,j) / n(i)n(j)]
        pmi = log((1.0 * num_window * count / (word_freq_i * word_freq_j)))
        if pmi <= 0:
            continue
        pmi_word_word[vocab[wi]][vocab[wj]] = pmi

    return pmi_word_word

if __name__ == "__main__":
    doc_content_list = []
    prj_path = "../temp/"
    dataset = "R8"
    f = open(prj_path + dataset + '.clean.txt', 'r')
    lines = f.readlines()
    for line in lines:
        doc_content_list.append(line.strip())
    f.close()
    tf_idf_matrix, sorted_tfvocab = get_tfidf_score(doc_content_list)
    pmi_matrix = get_pmi(prj_path, dataset, doc_content_list, list(sorted_tfvocab.keys()))
