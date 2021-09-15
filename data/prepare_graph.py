import os
import sys
sys.path.append("../")
from typing import Tuple, List, Dict

import numpy as np
import torch
from tqdm import tqdm
from dgl import DGLGraph

import utils
import data.utils as data_utils
import data.dataset as dataset


def prepare_graph(
        doc_path: str,
        cache_dir: str, 
        feature_type: List[str], 
        lang: str = "en",
        par_num: List[int] = None, 
        n_clusters: int = None,
    ) -> Tuple[DGLGraph, int]:
    os.makedirs(cache_dir, exist_ok=True)
    cache_doc_path = os.path.join(cache_dir, "cached_doc.pck")
    cache_word_path = os.path.join(cache_dir, "cached_word.pck")
    cache_cluster_path = os.path.join(cache_dir, "cached_cluster.pck")
    cache_concept_path = os.path.join(cache_dir, "cached_concept.pck")

    ds = dataset.PatentClassificationDataset(predict_category="doc")
    D_encoder, D_feat, D_label, D_mask, doc_content = utils.cache_to_path(
        cache_doc_path, get_document, doc_path, lang=lang, cache_dir=cache_dir
    )
    ds.add_nodes("doc", D_encoder, feat=D_feat, label=D_label, **D_mask)

    if "word" in feature_type:
        W_encoder, W_feat, DvsW_weight, WvsW_weight = utils.cache_to_path(
            cache_word_path, get_document_word, doc_content, lang=lang, cache_dir=cache_dir
        )
        DvsW = DvsW_weight.nonzero()
        WvsW = WvsW_weight.nonzero()
        ds.add_nodes("word", W_encoder, feat=W_feat)
        ds.add_edges(("word", "word#relate", "word"), WvsW, weight=WvsW_weight[WvsW])
        ds.add_edges(("doc", "word#have", "word"), DvsW, weight=DvsW_weight[DvsW])

    if "label_cluster" in feature_type or "doc_cluster" in feature_type:
        embedding = None if "label_cluster" in feature_type else D_feat
        Cl_encoder, Cl_feat, DvsCl_weight, ClvsCl_weight = utils.cache_to_path(
            cache_cluster_path, get_document_cluster, D_feat, embedding, n_clusters=n_clusters
        )
        DvsCl = DvsCl_weight.nonzero()
        ClvsCl = ClvsCl_weight.nonzero()
        ds.add_nodes("cluster", Cl_encoder, feat=Cl_feat)
        ds.add_edges(("cluster", "cluster#relate", "cluster"), ClvsCl, weight=ClvsCl_weight[ClvsCl])
        ds.add_edges(("doc", "cluster#form", "cluster"), DvsCl, weight=DvsCl_weight[DvsCl])

    if "concept" in feature_type:
        C_encoder, C_feat, DvsC, CvsC = utils.cache_to_path(
            cache_concept_path, get_document_concept, D, par_num=par_num, cache_dir=cache_dir
        )
        ds.add_nodes("concept", C_encoder, feat=C_feat)
        ds.add_edges(("concept", "concept#in", "concept"), CvsC)
        ds.add_edges(("doc", "concept#have", "concept"), DvsC)

    return ds.get_graph(), ds.predict_category, ds.num_classes
 
def get_document(doc_path: str, lang: str = "en", cache_dir: str = "cache/"):
    docs = {doc["id"] : doc for doc in utils.load_ndjson(doc_path)}
    D = {did: id for id, did in enumerate(sorted(docs.keys()))}
    label_encoder = set()
    doc_content = []
    D_feat, D_label = [], []
    D_mask = {dtype: [] for dtype in ("train_mask", "val_mask", "test_mask")}
    for did in tqdm(D, desc="Loading documents"):
        x = docs[did]
        label_encoder.update(x["labels"])
        doc_content.append(x["desc"][:30000])
        D_label.append(x["labels"])
        D_feat.append(x["desc"][:10000])
        D_mask["train_mask"].append(x["is_train"])
        D_mask["val_mask"].append(x["is_dev"])
        D_mask["test_mask"].append(x["is_test"])

    label_encoder = {lid: id for id, lid in enumerate(sorted(label_encoder))}
    utils.dump_json(label_encoder, os.path.join(cache_dir, "label_encoder.json"))
    D_label = utils.get_multihot_encoding(D_label, label_encoder)
    D_feat = utils.get_bert_features(D_feat, max_length=512, lang=lang)
    doc_content = utils.normalize_text(doc_content, lang=lang, cache_dir=cache_dir)

    return D, D_feat, D_label, D_mask, doc_content

def get_document_word(doc_content: List[str], lang: str = "en", cache_dir: str = "cache/"):
    DvsW_weight, W = utils.get_tfidf_score(doc_content, lang=lang, cache_dir=cache_dir)
    sorted_words = sorted(W, key=W.get)
    W_feat = utils.get_word_embedding(sorted_words, corpus=doc_content, cache_dir=cache_dir)
    WvsW_weight = utils.get_pmi(doc_content, vocab=sorted_words, window_size=20)
    DvsW_weight = DvsW_weight.toarray()
    WvsW_weight = np.tril(WvsW_weight)
    return W, W_feat, DvsW_weight, WvsW_weight

def get_document_cluster(D_feat: List, embedding: List = None, n_clusters: int = 100):
    if embedding is None:
        # subclass_titles = [e for x in data_utils.IPC_SUBCLASS.values() for e in x]
        subclass_titles = [e for x in data_utils.IPC_CLASS.values() for e in x]
        embedding = utils.get_bert_features(subclass_titles, max_length=32)
    Cl = {str(id): id for id in range(n_clusters)}
    Cl_feat = utils.get_kmean_matrix(embedding, n_clusters=n_clusters)
    ClvsCl_weight = np.tril(utils.pairwise_distances(Cl_feat, n_jobs=4))
    DvsCl_weight = utils.pairwise_distances(D_feat, Cl_feat, n_jobs=4)
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
