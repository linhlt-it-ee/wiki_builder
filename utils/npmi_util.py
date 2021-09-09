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
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def normalize_text(doc_content_list: List[str], lang: str = "en", cache_dir="./tmp"):
    norm_text_path = os.path.join(cache_dir, "norm_text.pck")
    os.makedirs(cache_dir, exist_ok=True)
    if os.path.exists(norm_text_path):
        with open(norm_text_path, "rb") as f:
            return pickle.load(f)

    res = []
    stemmer = porter.PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    tagger = fugashi.Tagger()
    for doc in tqdm(doc_content_list, desc="Normalizing"):
        if lang == "en":
            words = [w for w in nltk.wordpunct_tokenize(doc) if w.isalpha() and len(w) > 2]
            words = [lemmatizer.lemmatize(x, "v") for x in words]
            words = [lemmatizer.lemmatize(x, "n") for x in words]
            words = [stemmer.stem(x) for x in words]
        elif lang == "ja":
            words = [str(w.feature.lemma) for w in tagger(doc)]
            words = [w for w in words if w.isalpha() and w != "None"]
        else:
            raise NotImplementedError
        res.append(" ".join(words).lower())

    with open(norm_text_path, "wb") as f:
        pickle.dump(res, f)
    return res

def get_tfidf_score(texts, lang: str = "en", cache_dir="./tmp"):
    stop_words = "english" if lang == "en" else None
    vectorizer = TfidfVectorizer(tokenizer=lambda x : x.split(), stop_words=stop_words, min_df=10, max_df=0.7)
    X = vectorizer.fit_transform(texts)
    tf_vocab = vectorizer.vocabulary_

    print("Vocabulary size:", len(tf_vocab))
    os.makedirs(cache_dir, exist_ok=True)
    with open(os.path.join(cache_dir, "vocab.txt"), "w") as f:
        f.write(" ".join(sorted(tf_vocab.keys())))

    return X, tf_vocab

def get_pmi(doc_content_list: List[str], vocab: List[str], window_size: int = 20):
    vocab = {w : i for i, w in enumerate(vocab)}
    word_freq = defaultdict(lambda : 0)
    for doc_words in doc_content_list:
        for word in doc_words.split():
            if word in vocab:
                word_freq[word] += 1

    assert len(word_freq) == len(vocab), f"Vocab size mismatch: {len(word_freq)} != {len(vocab)}"
    word_doc_list = defaultdict(list)
    for i in range(len(doc_content_list)):
        doc_words = doc_content_list[i]
        words = doc_words.split()
        appeared = set()
        for word in doc_words.split():
            if word in appeared or word not in vocab:
                continue
            if word in word_doc_list:
                word_doc_list[word].append(i)
            appeared.add(word)

    word_doc_freq = {}
    for word, doc_list in word_doc_list.items():
        word_doc_freq[word] = len(doc_list)

    # word co-occurence within context windows
    windows = []
    for doc_words in tqdm(doc_content_list, desc="Striding"):
        words = [x for x in doc_words.split() if x in vocab]
        length = len(words)
        if length <= window_size:
            windows.append(words)
        else:
            # print(length, length - window_size + 1)
            for j in range(length - window_size + 1):
                window = words[j: j + window_size]
                windows.append(window)

    word_window_freq = defaultdict(lambda : 0)
    for window in tqdm(windows, desc="Counting words within window"):
        appeared = set()
        for i in range(len(window)):
            if window[i] in appeared:
                continue
            word_window_freq[window[i]] += 1
            appeared.add(window[i])

    word_pair_count = defaultdict(lambda : 0)
    for window in tqdm(windows, desc="Counting word pairs within window"):
        for i in range(1, len(window)):
            for j in range(i):
                wi, wj = window[i], window[j]
                if wi == wj:
                    continue
                # two orders
                word_pair_count[(wi, wj)] += 1
                word_pair_count[(wj, wi)] += 1

    # pmi as weights
    num_window = len(windows)
    pmi_word_word = np.zeros((len(vocab), len(vocab)))
    for (wi, wj), count in word_pair_count.items():
        word_freq_i = word_window_freq[wi]
        word_freq_j = word_window_freq[wj]
        pmi = log((1.0 * count / num_window) /
                  (1.0 * word_freq_i * word_freq_j / (num_window * num_window)))
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
