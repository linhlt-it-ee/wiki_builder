import os
from typing import List

import fugashi
import nltk
from nltk.stem import WordNetLemmatizer, porter
from tqdm import tqdm

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
            # words = [str(w.feature.lemma) for w in tagger(doc)]
            words = [str(w.surface) for w in tagger(doc)]
            words = [w for w in words if w.isalpha() and w != "None"]
        return " ".join(words).lower()

    return fn


def normalize_text(doc_content_list: List[str], lang: str = "en", cache_dir="./tmp"):
    transformer = normalizer(lang=lang)
    res = [transformer(doc) for doc in tqdm(doc_content_list, desc="Normalizing")]
    file_utils.dump(res, os.path.join(cache_dir, f"normalized_text_{lang}.pck"))
    return res
