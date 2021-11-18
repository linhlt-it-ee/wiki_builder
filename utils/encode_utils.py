import os
import logging
from typing import List, Dict

import nltk
import torch
import numpy as np
import fugashi
import gensim.downloader
from gensim.models import Word2Vec
from tqdm import tqdm
from nltk.stem import porter, WordNetLemmatizer
from transformers import AutoModel, AutoTokenizer
from sklearn.preprocessing import MultiLabelBinarizer

from . import file_utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def encode_multihot(labels):
    mlb = MultiLabelBinarizer()
    res = mlb.fit_transform(labels)
    return res, list(mlb.classes_)

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

def encode_bert(text: List[str], max_length: int = 64, lang: str = "en"):
    if lang == "en":
        pretrained_model_name = "distilbert-base-uncased"
    elif lang == "ja":
        pretrained_model_name = "cl-tohoku/bert-base-japanese-char"
    else:
        raise NotImplementedError

    model = AutoModel.from_pretrained(pretrained_model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    model.eval()
    res = []
    for batch_text in yield_batch(text):
        inputs = tokenizer.batch_encode_plus(
            batch_text, padding="max_length", truncation=True, 
            max_length=max_length, return_tensors="pt"
        ).to(device)
        hidden_state = model(**inputs).last_hidden_state.detach().cpu().numpy()
        embedding = np.take(hidden_state, indices=0, axis=1)    # pool first
        res.append(embedding)

    return np.vstack(res)

def encode_sbert(text: List[str], lang: str = "en"):
    from sentence_transformers import SentenceTransformer
    if lang == "en":
        pretrained_model_name = "sentence-transformers/all-distilroberta-v1"
    else:
        raise NotImplementedError

    model = SentenceTransformer(pretrained_model_name)
    res = []
    for batch_text in yield_batch(text):
        embedding = model.encode(batch_text)
        res.append(embedding)
    return np.vstack(res)

def yield_batch(arr: List):
    arr_len = len(arr)
    pbar = tqdm(total=arr_len, desc="Batching")
    start_idx = 0
    while start_idx < arr_len - 1:
        batch_size = min(32, arr_len - start_idx)
        yield arr[start_idx : start_idx + batch_size]
        pbar.update(batch_size)
        start_idx += batch_size

def encode_word(words: List[str], corpus: List[str], cache_dir="./tmp"):
    sentences = [sent.split() for sent in corpus]
    model_path = os.path.join(cache_dir, "word2vec.model")
    if not os.path.exists(model_path): 
        # same dimension as bert embedding
        model = Word2Vec(sentences=sentences, vector_size=768, window=5, workers=4)
        model.save(model_path)
    else:
        model = Word2Vec.load(model_path)

    return [model.wv[w] for w in tqdm(words, desc="Featurizing")]