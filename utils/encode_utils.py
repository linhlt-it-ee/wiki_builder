import os
from typing import List

import numpy as np
import torch
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from .helper import yield_batch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def encode_multihot(labels):
    mlb = MultiLabelBinarizer()
    res = mlb.fit_transform(labels)
    return res, list(mlb.classes_)


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
            batch_text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        hidden_state = model(**inputs).last_hidden_state.detach().cpu().numpy()
        embedding = np.take(hidden_state, indices=0, axis=1)  # pool first
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


def encode_tfidf(text: List[str], vocab: List[str] = None, lang: str = "en", cache_dir="./tmp"):
    stop_words = None if lang != "en" else "english"
    vectorizer = TfidfVectorizer(
        tokenizer=lambda x: x.split(), stop_words=stop_words, vocabulary=None, min_df=5, max_df=0.7
    ).fit(text)
    tf_vocab = vectorizer.vocabulary_
    if vocab is not None:
        vocab = list(set(vocab).intersection(tf_vocab.keys()))
        assert len(vocab) != 0, "Empty vocabulary"
        vectorizer = TfidfVectorizer(
            tokenizer=lambda x: x.split(),
            stop_words=stop_words,
            vocabulary=vocab,
            min_df=5,
            max_df=0.7,
        ).fit(text)
        tf_vocab = vectorizer.vocabulary_
    X = vectorizer.transform(text)

    print("Vocabulary size:", len(tf_vocab))
    with open(os.path.join(cache_dir, f"vocab_{lang}.txt"), "w") as f:
        f.write(" ".join(sorted(tf_vocab.keys())))

    return X, tf_vocab
