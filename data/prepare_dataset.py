import os
import sys
sys.path.append("../")
from typing import Tuple, List, Dict
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm
from dgl import DGLGraph
from sklearn.preprocessing import normalize

import utils
import data.utils as data_utils
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
    D_encoder, D_feat, (D_label, label_encoder), (D_aux_label, aux_label_encoder), D_mask, doc_content, doc_labels = utils.cache_to_path(
        cache_path, get_document, doc_path, lang=lang, cache_dir=cache_dir
    )
    ds.add_nodes("doc", D_encoder, feat=D_feat, **D_mask)
    ds.add_labels(D_label, label_encoder)

    if "word" in feature_type:
        cache_path = os.path.join(cache_dir, "cached_word.pck")
        W_encoder, W_feat, DvsW_weight, WvsW_weight = utils.cache_to_path(
            cache_path, get_document_word, doc_content, lang=lang, cache_dir=cache_dir
        )
        DvsW = DvsW_weight.nonzero()
        WvsW = WvsW_weight.nonzero()
        ds.add_nodes("word", W_encoder, feat=W_feat)
        ds.add_edges(("word", "word#relate", "word"), WvsW, weight=WvsW_weight[WvsW])
        ds.add_edges(("doc", "word#have", "word"), DvsW, weight=DvsW_weight[DvsW])

    if "label" in feature_type:
        cache_path = os.path.join(cache_dir, "cached_label.pck")
        L_encoder, L_feat, DvsL_weight = utils.cache_to_path(
            cache_path, get_document_label, D_feat, doc_labels
        )
        DvsL = DvsL_weight.nonzero()
        ds.add_nodes("label", L_encoder, feat=L_feat)
        ds.add_edges(("doc", "label#distance", "label"), DvsL, weight=DvsL_weight[DvsL])

    if "cluster" in feature_type:
        cache_path = os.path.join(cache_dir, "cached_cluster.pck")
        Cl_encoder, Cl_feat, DvsCl_weight, ClvsCl_weight = utils.cache_to_path(
            cache_path, get_document_cluster, D_feat, n_clusters=n_clusters
        )
        DvsCl = DvsCl_weight.nonzero()
        ClvsCl = ClvsCl_weight.nonzero()
        ds.add_nodes("cluster", Cl_encoder, feat=Cl_feat)
        ds.add_edges(("cluster", "cluster#distance", "cluster"), ClvsCl, weight=ClvsCl_weight[ClvsCl])
        ds.add_edges(("doc", "cluster#form", "cluster"), DvsCl, weight=DvsCl_weight[DvsCl])

    if "concept" in feature_type:
        cache_path = os.path.join(cache_dir, "cached_concept.pck")
        C_encoder, C_feat, DvsC, CvsC = utils.cache_to_path(
            cache_path, get_document_concept, D_encoder.values(), par_num=par_num, cache_dir=cache_dir
        )
        ds.add_nodes("concept", C_encoder, feat=C_feat)
        ds.add_edges(("concept", "concept#is-child", "concept"), CvsC)
        ds.add_edges(("doc", "concept#have", "concept"), DvsC)

    return ds
 
def get_document(doc_path: str, lang: str = "en", cache_dir: str = "cache/"):
    docs = {doc["id"]: doc for doc in utils.load_ndjson(doc_path)}
    D = {did: id for id, did in enumerate(sorted(docs.keys()))}
    D_mask = {dtype: [] for dtype in ("train_mask", "val_mask", "test_mask")}
    doc_content, doc_labels = [], []
    doc_aux_labels = []
    for did in tqdm(D, desc="Loading documents"):
        x = docs[did]
        content = " ".join((x["abstract"], x["title"], x["claim_1"], x["description"]))
        doc_content.append(content[:5000])
        doc_labels.append(x["labels"])
        doc_aux_labels.append([e[:3] for e in x["labels"]])
        D_mask["train_mask"].append(x["is_train"])
        D_mask["val_mask"].append(x["is_dev"])
        D_mask["test_mask"].append(x["is_test"])

    label_encoder = set([e for x in doc_labels for e in x])
    label_encoder = {lid: id for id, lid in enumerate(sorted(label_encoder))}
    utils.dump_json(label_encoder, os.path.join(cache_dir, "label_encoder.json"))
    D_feat = utils.get_sbert_embedding(doc_content, lang=lang)
    D_label = utils.get_multihot_encoding(doc_labels, label_encoder)
    return D, D_feat, (D_label, label_encoder), D_mask, doc_content, doc_labels

def get_document_word(doc_content: List[str], lang: str = "en", cache_dir: str = "cache/"):
    label_descriptions = [e for x in data_utils.IPC_SUBCLASS.values() for e in x]
    label_descriptions = utils.normalize_text(label_descriptions, lang="en", cache_dir="./")
    _, vocab = utils.get_tfidf_score(label_descriptions, lang="en", cache_dir="./")
    doc_content = utils.normalize_text(doc_content, lang=lang, cache_dir=cache_dir)
    DvsW_weight, W = utils.get_tfidf_score(doc_content, vocab=vocab, lang=lang, cache_dir=cache_dir)
    sorted_words = sorted(W, key=W.get)
    W_feat = utils.get_word_embedding(sorted_words, corpus=doc_content, cache_dir=cache_dir)
    WvsW_weight = utils.get_pmi(doc_content, vocab=sorted_words, window_size=20)
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

    C, CvsC_edges, DvsC_edges = data_utils.create_document_concept_graph(D.keys(), doc_mention, mention_concept, par_num)
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
