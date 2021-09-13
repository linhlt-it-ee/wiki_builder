import os
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

from . import file_utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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

def get_multihot_encoding(labels: List[List[str]], label_encoder: Dict[str, int]):
    n_samples = len(labels)
    embedding = np.zeros((n_samples, len(label_encoder)))
    slice_idx = np.array([[rid, label_encoder[e]] for rid, x in enumerate(labels) for e in x]).T.tolist()
    embedding[slice_idx] = 1
    return embedding

def get_bert_features(text: List[str], max_length: int = 64, lang: str = "en"):
    if lang == "en":
        pretrained_model_name = "distilbert-base-uncased"
    elif lang == "ja":
        pretrained_model_name = "cl-tohoku/bert-base-japanese-char"
    else:
        raise NotImplementedError
    model = AutoModel.from_pretrained(pretrained_model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

    res = []
    n_text = len(text)
    pbar = tqdm(total=n_text, desc="Featurizing")
    model.eval()
    start_idx = 0
    with torch.no_grad():
        while start_idx < n_text - 1:
            batch_size = min(32, n_text - start_idx)
            # truncate to prevent segmentation fault and speed up
            batch_text = [x[:5000] for x in text[start_idx : start_idx + batch_size]]
            inputs = tokenizer.batch_encode_plus(
                batch_text,
                padding="max_length", truncation=True, 
                max_length=max_length, return_tensors="pt"
            ).to(device)
            hidden_state = model(**inputs).last_hidden_state.detach().cpu().numpy()
            embedding = np.take(hidden_state, indices=0, axis=1)    # pool first
            res.append(embedding)
            pbar.update(batch_size)
            start_idx += batch_size

    return np.vstack(res)

def get_word_embedding(words: List[str], corpus: List[str], cache_dir="./tmp"):
    sentences = [sent.split() for sent in corpus]
    model_path = os.path.join(cache_dir, "word2vec.model")
    if not os.path.exists(model_path): 
        # same dimension as bert embedding
        model = Word2Vec(sentences=sentences, vector_size=768, window=5, workers=4)
        model.save(model_path)
    else:
        model = Word2Vec.load(model_path)

    return [model.wv[w] for w in tqdm(words, desc="Featurizing")]

def get_phrase_embedding(phrases: List[str]):
    model = gensim.downloader.load("glove-wiki-gigaword-300")
    res = []
    for p in tqdm(phrases, desc="Featurizing"):
        embedding = np.array([model[w] for w in p.split() if w in model]).mean(axis=0)
        if type(embedding) == np.float64:
            embedding = np.zeros((300,))
        embedding = np.concatenate((embedding, np.zeros(768 - 300)))
        assert embedding.shape == (768,), f"{embedding.shape}"
        res.append(embedding)
    return res
