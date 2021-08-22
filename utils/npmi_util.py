import os
from typing import List
from math import log
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from nltk.stem import porter
from sklearn.feature_extraction.text import TfidfVectorizer

stemmer = porter.PorterStemmer()

def stem_text(doc_content_list: List[str]):
    res = []
    for doc in tqdm(doc_content_list, desc="Stemming"):
        res.append(" ".join([stemmer.stem(x) for x in doc.split()]))
    return res

def split_tokenizer(text):
    return text.split()

def get_pmi(doc_content_list: List[str], vocab: List[str], cache_dir: str = "./tmp"):
    # build vocab
    with open(os.path.join(cache_dir, "vocab.txt"), "w") as f:
        f.write("\n".join(vocab))

    vocab = set(vocab)
    word_freq = defaultdict(lambda : 0)
    for doc_words in doc_content_list:
        for word in doc_words.split():
            if word in vocab:
                word_freq[word] += 1

    assert len(word_freq) == len(vocab)
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

    # word co-occurence with context windows
    window_size = 20
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
                # print(window)

    word_window_freq = defaultdict(lambda : 0)
    for window in tqdm(windows, desc="Counting words within window"):
        appeared = set()
        for i in range(len(window)):
            if window[i] in appeared:
                continue
            word_window_freq[window[i]] += 1
            appeared.add(window[i])

    vocab = list(vocab)
    vocab_size = len(vocab)
    word_id_map = {}
    for i in range(vocab_size):
        word_id_map[vocab[i]] = i

    word_pair_count = defaultdict(lambda : 0)
    for window in tqdm(windows, desc="Counting word pairs within window"):
        for i in range(1, len(window)):
            for j in range(i):
                word_i = window[i]
                word_i_id = word_id_map[word_i]
                word_j = window[j]
                word_j_id = word_id_map[word_j]
                if word_i_id == word_j_id:
                    continue
                # two orders
                word_pair_str = str(word_i_id) + ',' + str(word_j_id)
                word_pair_count[word_pair_str] += 1
                word_pair_str = str(word_j_id) + ',' + str(word_i_id)
                word_pair_count[word_pair_str] += 1

    # pmi as weights
    num_window = len(windows)
    pmi_word_word = np.zeros((len(vocab),len(vocab)))
    for key in word_pair_count:
        temp = key.split(',')
        i = int(temp[0])
        j = int(temp[1])
        count = word_pair_count[key]
        word_freq_i = word_window_freq[vocab[i]]
        word_freq_j = word_window_freq[vocab[j]]
        pmi = log((1.0 * count / num_window) /
                  (1.0 * word_freq_i * word_freq_j / (num_window * num_window)))
        if pmi <= 0:
            continue
        pmi_word_word[i][j] = pmi
        # row.append(train_size + i)
        # col.append(train_size + j)
        # weight.append(pmi)
    # print("word pair count",word_pair_count_wth_id)

    # print("word id map", len(word_id_map.keys()))
    return pmi_word_word

def get_tfidf_score(texts):
    vectorizer = TfidfVectorizer(tokenizer=split_tokenizer, stop_words="english", min_df=5, max_df=0.5)
    X = vectorizer.fit_transform(texts)
    tf_vocab = vectorizer.vocabulary_
    sorted_tfvocab = {k: v for k, v in sorted(tf_vocab.items(), key=lambda item: item[1])}
    print("Vocabulary size:", len(sorted_tfvocab))
    return X, sorted_tfvocab

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
