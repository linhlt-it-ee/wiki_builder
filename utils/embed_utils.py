import os
from typing import List, Dict, Tuple

import torch
import numpy as np
from gensim.models import Word2Vec
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_onehot(labels: List[str], label_ids: Dict[str, int]):
    embed = [0] * len(label_ids)
    for x in labels:
        embed[label_ids[x]] = 1
    return embed

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
    start_idx = last_idx = 0
    with torch.no_grad():
        while start_idx < n_text - 1:
            batch_size = min(32, n_text - start_idx)
            # prevent seg fault and speed up time
            batch_text = [x[:5000] for x in text[start_idx : start_idx + batch_size]]
            inputs = tokenizer.batch_encode_plus(
                batch_text,
                padding="max_length", 
                truncation=True, 
                max_length=max_length, 
                return_tensors="pt"
            ).to(device)
            hidden_state = model(**inputs).last_hidden_state.detach().cpu().numpy()
            embedding = np.take(hidden_state, indices=0, axis=1)
            res.append(embedding)
            pbar.update(batch_size)
            start_idx += batch_size

    return np.vstack(res)

def get_word_embedding(words: List[str], corpus: List[str], cache_dir="./tmp"):
    sentences = [sent.split() for sent in corpus]
    os.makedirs(cache_dir, exist_ok=True)
    model_path = os.path.join(cache_dir, "word2vec.model")
    if not os.path.exists(model_path): 
        # same dimension as bert embedding
        model = Word2Vec(sentences=sentences, vector_size=768, window=5, workers=4)
        model.save(model_path)
    else:
        model = Word2Vec.load(model_path)

    return [model.wv[w] for w in tqdm(words, desc="Featurizing")]
