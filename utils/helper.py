from collections import defaultdict
from math import log
from typing import List

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.mixture import GaussianMixture
from tqdm import tqdm


def yield_batch(arr: List, batch_size: int = 32):
    arr_len = len(arr)
    pbar = tqdm(total=arr_len, desc="Batching")
    start_idx = 0
    while start_idx <= arr_len - 1:
        batch_size = min(batch_size, arr_len - start_idx)
        yield arr[start_idx : start_idx + batch_size]
        pbar.update(batch_size)
        start_idx += batch_size


def get_kmean_matrix(embeddings: List, n_clusters: int = 100, is_gmm: bool = False):
    if not is_gmm:
        model = KMeans(n_clusters=n_clusters).fit(embeddings)
        centers = model.cluster_centers_
    else:
        model = GaussianMixture(
            n_components=n_clusters,
            covariance_type="full",
            random_state=0,
        ).fit(embeddings)
        centers = model.means_

    return centers


def get_pmi(doc_content_list: List[str], vocab: List[str], window_size: int = 20):
    vocab = {w: i for i, w in enumerate(vocab)}
    doc_word_list = [[w for w in x.split() if w in vocab] for x in doc_content_list]
    windows = []
    for words in doc_word_list:
        length = len(words)
        if length <= window_size:
            windows.append(list(set(words)))
        else:
            for j in range(length - window_size + 1):
                window = list(set(words[j : j + window_size]))
                windows.append(window)

    word_window_freq = defaultdict(lambda: 0)
    for window in tqdm(windows, desc="Word frequency (windows)"):
        for w in window:
            word_window_freq[w] += 1

    word_pair_count = defaultdict(lambda: 0)
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
