import os
from typing import List
from math import log
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer

def get_tfidf_score(text: List[str], vocab: List[str] = None, lang: str = "en", cache_dir="./tmp"):
    stop_words = None if lang != "en" else "english"
    vectorizer = TfidfVectorizer(
        tokenizer=lambda x : x.split(), stop_words=stop_words, vocabulary=None,
        min_df=5, max_df=0.7
    ).fit(text)
    tf_vocab = vectorizer.vocabulary_
    if vocab is not None:
        vocab = list(set(vocab).intersection(tf_vocab.keys()))
        assert len(vocab) != 0, "Empty vocabulary"
        vectorizer = TfidfVectorizer(
            tokenizer=lambda x : x.split(), stop_words=stop_words, vocabulary=vocab,
            min_df=5, max_df=0.7
        ).fit(text)
        tf_vocab = vectorizer.vocabulary_
    X = vectorizer.transform(text)
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
            windows.append(list(set(words)))
        else:
            for j in range(length - window_size + 1):
                window = list(set(words[j: j + window_size]))
                windows.append(window)

    word_window_freq = defaultdict(lambda : 0)
    for window in tqdm(windows, desc="Word frequency (windows)"):
        for w in window:
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
        # pmi = log[p(i,j) / p(i)p(j)] = log[n * n(i,j) / n(i)n(j)]
        pmi = log((1.0 * num_window * count / (word_window_freq[wi] * word_window_freq[wj])))
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
