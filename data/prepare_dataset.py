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
        ds.add_edges(("doc", "word#have", "word"), DvsW, weight=DvsW_weight[DvsW])

    if "cluster" in feature_type:
        cache_path = os.path.join(cache_dir, "cached_cluster.pck")
        Cl, Cl_feat, DvsCl_weight, ClvsCl_weight = utils.cache_to_path(
            cache_path,
            get_document_cluster,
            D_feat,
            n_clusters=n_clusters,
        )
        DvsCl = DvsCl_weight.nonzero()
        ClvsCl = ClvsCl_weight.nonzero()
        ds.add_nodes("cluster", Cl, feat=Cl_feat)
        ds.add_edges(
            ("cluster", "cluster#distance", "cluster"), ClvsCl, weight=ClvsCl_weight[ClvsCl]
        )
        ds.add_edges(("doc", "cluster#form", "cluster"), DvsCl, weight=DvsCl_weight[DvsCl])

    if "label" in feature_type:
        cache_path = os.path.join(cache_dir, "cached_label.pck")
        L_encoder, L_feat, DvsL_weight = utils.cache_to_path(
            cache_path, get_document_label, D_feat, doc_labels
        )
        DvsL = DvsL_weight.nonzero()
        ds.add_nodes("label", L_encoder, feat=L_feat)
        ds.add_edges(("doc", "label#distance", "label"), DvsL, weight=DvsL_weight[DvsL])

    if "concept" in feature_type:
        cache_path = os.path.join(cache_dir, "cached_concept.pck")
        C_encoder, C_feat, DvsC, CvsC = utils.cache_to_path(
            cache_path,
            get_document_concept,
            D_encoder.values(),
            par_num=par_num,
            cache_dir=cache_dir,
        )
        ds.add_nodes("concept", C_encoder, feat=C_feat)
        ds.add_edges(("concept", "concept#is-child", "concept"), CvsC)
        ds.add_edges(("doc", "concept#have", "concept"), DvsC)

    return ds


def get_document(doc_path: str, lang: str = "en", cache_dir: str = "cache/"):
    docs = {doc["id"]: doc for doc in utils.load_ndjson(doc_path)}
    D = sorted(docs.keys())
    D_mask = {dtype: [] for dtype in ("train_mask", "val_mask", "test_mask")}
    doc_content, doc_labels = [], []
    for did in tqdm(D, desc="Loading documents"):
        doc = docs[did]
        doc_content.append(doc["text"])
        doc_labels.append(doc["labels"])
        D_mask["train_mask"].append(doc["is_train"])
        D_mask["val_mask"].append(doc["is_dev"])
        D_mask["test_mask"].append(doc["is_test"])

    D_label, classes = utils.encode_multihot(doc_labels)
    D_feat = utils.encode_sbert(doc_content, lang=lang)
    return D, D_feat, (classes, D_label), D_mask


def get_document_word(doc_path: str, D: List, lang: str = "en", cache_dir: str = "cache/"):
    doc_content = [None] * len(D)
    D_mapper = {did: id for id, did in enumerate(D)}
    for doc in utils.load_ndjson(doc_path):
        doc_content[D_mapper[doc["id"]]] = doc["text"]
    doc_content = utils.normalize_text(doc_content, lang=lang, cache_dir=cache_dir)

    # extract key words from class description
    if lang == "en":
        class_desc = [e for x in data_utils.CPC_SUBCLASS.values() for e in x]
        class_desc = utils.normalize_text(class_desc, lang="en")
        _, class_W = utils.encode_tfidf(class_desc, lang="en")
    else:
        class_W = None

    DvsW_weight, W = utils.encode_tfidf(doc_content, vocab=class_W, lang=lang, cache_dir=cache_dir)
    W_feat = utils.encode_word(W, corpus=doc_content, cache_dir=cache_dir)
    WvsW_weight = utils.get_pmi(doc_content, vocab=W, window_size=10)
    DvsW_weight = DvsW_weight.toarray()
    WvsW_weight = np.tril(WvsW_weight)
    return W, W_feat, DvsW_weight, WvsW_weight


def get_document_label(D_feat: str, doc_labels: List[List[str]]):
    tmp_path = "./label_embedding.pck"
    label_encoder = defaultdict(list)
    desc_embeddings = []
    for label, descriptions in data_utils.IPC_SUBCLASS.items():
        for desc in descriptions:
            label_encoder[label].append(len(desc_embeddings))
            desc_embeddings.append(desc)
    if os.path.exists(tmp_path):
        desc_embeddings = utils.load(tmp_path)
    else:
        desc_embeddings = utils.get_sbert_embedding(desc_embeddings, lang="en")
        utils.dump(desc_embeddings, tmp_path)

    num_labels = len(data_utils.IPC_SUBCLASS)
    L = {str(id): id for id in range(num_labels)}
    L_feat, DvsL_weight = [], []
    for label, descriptions in tqdm(data_utils.IPC_SUBCLASS.items(), desc="Label embedding"):
        inputs = [desc_embeddings[desc_id] for desc_id in label_encoder[label]]
        feat = utils.get_kmean_matrix(inputs, n_clusters=1)
        dist = utils.pairwise_distances(D_feat, inputs, n_jobs=4).min(axis=1)
        # dist = utils.pairwise_distances(D_feat, feat, n_jobs=4).squeeze()
        L_feat.append(feat)
        DvsL_weight.append(dist)
    L_feat = np.vstack(L_feat).astype(np.float32)
    DvsL_weight = normalize(np.vstack(DvsL_weight).T)
    DvsL_weight.partition(200, axis=1)
    DvsL_weight = DvsL_weight[:, :200]
    return L, L_feat, DvsL_weight


def get_document_cluster(D_feat: List, n_clusters: int = 100):
    Cl = {str(id): id for id in range(n_clusters)}
    Cl_feat = utils.get_kmean_matrix(D_feat, n_clusters=n_clusters)
    ClvsCl_weight = np.tril(normalize(utils.pairwise_distances(Cl_feat, n_jobs=4)))
    DvsCl_weight = normalize(utils.pairwise_distances(D_feat, Cl_feat, n_jobs=4))
    return Cl, Cl_feat, DvsCl_weight, ClvsCl_weight


def get_document_concept(D: Dict[str, int], par_num: List[int], cache_dir: str = "cache/"):
    # temporary files (for `concept` nodes)
    doc_mention_path = os.path.join(cache_dir, "doc_mention.json")
    mention_concept_path = os.path.join(cache_dir, "concept_links.json")
    concept_path = os.path.join(cache_dir, "concept_labels.json")
    doc_mention = utils.load_json(doc_mention_path)
    mention_concept = utils.load_json(mention_concept_path)

    C, CvsC_edges, DvsC_edges = data_utils.create_document_concept_graph(
        D.keys(), doc_mention, mention_concept, par_num
    )
    C = {cid: id for id, cid in enumerate(sorted(C))}
    concepts = utils.load_json(concept_path)
    C_feat = [concepts[cid] for cid in C]
    C_feat = utils.get_bert_features(C_feat, max_length=32)
    DvsC = ([], [])
    CvsC = ([], [])
    for u, v in DvsC_edges:
        DvsC[0].append(D[u])
        DvsC[1].append(C[v])
    for u, v in CvsC_edges:
        CvsC[0].append(C[u])
        CvsC[1].append(C[v])

    return C, C_feat, DvsC, CvsC
