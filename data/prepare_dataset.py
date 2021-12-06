import os
import sys

sys.path.append("../")
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from dgl import DGLGraph
from sklearn.preprocessing import normalize
from tqdm import tqdm

import data.utils as data_utils
import utils
from data.dataset import PatentClassificationDataset


def prepare_dataset(
    doc_path: str,
    cache_dir: str,
    feature_type: List[str],
    lang: str = "en",
    par_num: List[int] = None,
    n_clusters: int = None,
) -> PatentClassificationDataset:
    os.makedirs(cache_dir, exist_ok=True)
    ds = PatentClassificationDataset(predict_category="doc")

    cache_path = os.path.join(cache_dir, "cached_doc.pck")
    D, D_feat, (classes, D_label), D_mask = utils.cache_to_path(
        cache_path,
        get_document,
        doc_path,
        lang=lang,
        cache_dir=cache_dir,
    )
    ds.add_nodes("doc", D, feat=D_feat, **D_mask)
    ds.add_labels(classes, D_label)

    if "word" in feature_type:
        cache_path = os.path.join(cache_dir, "cached_word.pck")
        W, W_feat, DvsW_weight, WvsW_weight = utils.cache_to_path(
            cache_path,
            get_document_word,
            doc_path,
            D,
            lang=lang,
            cache_dir=cache_dir,
        )
        DvsW = DvsW_weight.nonzero()
        WvsW = WvsW_weight.nonzero()
        ds.add_nodes("word", W, feat=W_feat)
        ds.add_edges(("word", "word#relate", "word"), WvsW, weight=WvsW_weight[WvsW])
        ds.add_rev_edges(("doc", "word#have", "word"), DvsW, weight=DvsW_weight[DvsW])

    if "cluster" in feature_type:
        cache_path = os.path.join(cache_dir, "cached_cluster.pck")
        Cl, Cl_feat, DvsCl_weight, ClvsCl_weight = utils.cache_to_path(
            cache_path,
            get_document_cluster,
            D_feat,
            n_clusters=n_clusters,
            lang=lang,
            cache_dir=cache_dir,
        )
        DvsCl = DvsCl_weight.nonzero()
        ClvsCl = ClvsCl_weight.nonzero()
        ds.add_nodes("cluster", Cl, feat=Cl_feat)
        ds.add_rev_edges(("doc", "cluster#form", "cluster"), DvsCl, weight=DvsCl_weight[DvsCl])

    if "label" in feature_type:
        cache_path = os.path.join(cache_dir, "cached_label.pck")
        L, L_feat, DvsL_weight = utils.cache_to_path(
            cache_path,
            get_document_label,
            D_feat,
            lang=lang,
            cache_dir=cache_dir,
        )
        DvsL = DvsL_weight.nonzero()
        ds.add_nodes("label", L, feat=L_feat)
        ds.add_rev_edges(("doc", "label#distance", "label"), DvsL, weight=DvsL_weight[DvsL])

    return ds


def get_document(doc_path: str, lang: str = "en", cache_dir: str = "cache/"):
    docs = {doc["id"]: doc for doc in utils.load_ndjson(doc_path)}
    D = sorted(docs.keys())
    D_mask = {dtype: [] for dtype in ("train_mask", "val_mask", "test_mask")}
    doc_content, doc_labels = [], []
    for did in tqdm(D, desc="Loading documents"):
        doc = docs[did]
        doc_content.append(doc["text"][:20000])
        doc_labels.append(doc["labels"])
        D_mask["train_mask"].append(doc["is_train"])
        D_mask["val_mask"].append(doc["is_dev"])
        D_mask["test_mask"].append(doc["is_test"])

    D_label, classes = utils.encode_multihot(doc_labels)
    # D_feat = utils.encode_bert(doc_content, max_length=256, lang=lang)
    D_feat = utils.encode_sbert(doc_content, lang=lang)
    return D, D_feat, (classes, D_label), D_mask


def get_document_word(doc_path: str, D: List, lang: str = "en", cache_dir: str = "cache/"):
    doc_content = [None] * len(D)
    D_mapper = {did: id for id, did in enumerate(D)}
    for doc in utils.load_ndjson(doc_path):
        doc_content[D_mapper[doc["id"]]] = doc["text"][:20000]
    doc_content = utils.normalize_text(doc_content, lang=lang, cache_dir=cache_dir)

    # extract key words from class description
    if lang == "en":
        class_desc = [e for x in data_utils.CPC_SUBCLASS.values() for e in x]
        class_desc = utils.normalize_text(class_desc, lang="en")
        _, class_W = utils.encode_tfidf(class_desc, lang="en")
    else:
        class_W = None

    DvsW_weight, W = utils.encode_tfidf(
        doc_content, vocab=class_W, lang=lang, cache_dir=cache_dir, max_features=5000
    )
    vector_dim = 512 if lang == "ja" else 768
    W_feat = utils.encode_word(W, corpus=doc_content, dim=vector_dim, cache_dir=cache_dir)
    WvsW_weight = utils.get_pmi(doc_content, vocab=W, window_size=10)
    DvsW_weight = DvsW_weight.toarray()
    WvsW_weight = np.tril(WvsW_weight)
    return W, W_feat, DvsW_weight, WvsW_weight


def get_document_cluster(D_feat: List, n_clusters: int = 100):
    Cl = {str(id): id for id in range(n_clusters)}
    Cl_feat = utils.get_kmean_matrix(D_feat, n_clusters=n_clusters)
    ClvsCl_weight = np.tril(normalize(utils.pairwise_distances(Cl_feat, metric="cosine", n_jobs=4)))
    DvsCl_weight = normalize(utils.pairwise_distances(D_feat, Cl_feat, metric="cosine", n_jobs=4))
    return Cl, Cl_feat, DvsCl_weight, ClvsCl_weight


def get_document_label(D_feat: List, lang: str = "en", cache_dir: str = "cache/"):
    saved_path = os.path.join(cache_dir, "label_embedding.pck")
    desc_by_label = defaultdict(list)
    desc_embeddings = []
    for label, descriptions in data_utils.CPC_SUBCLASS.items():
        for desc in descriptions:
            desc_by_label[label].append(len(desc_embeddings))
            desc_embeddings.append(desc)
    if os.path.exists(saved_path):
        desc_embeddings = utils.load(saved_path)
    else:
        desc_embeddings = utils.encode_sbert(desc_embeddings, lang=lang)
        utils.dump(desc_embeddings, saved_path)

    num_labels = len(data_utils.CPC_SUBCLASS)
    L = {str(id): id for id in range(num_labels)}
    L_feat, DvsL_weight = [], []
    for label, descriptions in tqdm(
        data_utils.CPC_SUBCLASS.items(), desc="Computing label-document edge weight"
    ):
        inputs = [desc_embeddings[desc_id] for desc_id in desc_by_label[label]]
        feat = utils.get_kmean_matrix(inputs, n_clusters=1).astype(np.float32)
        # edge weight from a document to a representative label = minimum distance from document embedding to label description embedding
        dist = utils.pairwise_distances(D_feat, inputs, n_jobs=4).min(axis=1)
        # dist = utils.pairwise_distances(D_feat, feat, n_jobs=4).squeeze()
        L_feat.append(feat)
        DvsL_weight.append(dist)
    L_feat = np.vstack(L_feat)
    DvsL_weight = normalize(np.vstack(DvsL_weight).T)
    DvsL_weight.partition(200, axis=1)
    DvsL_weight = DvsL_weight[:, :200]
    return L, L_feat, DvsL_weight
