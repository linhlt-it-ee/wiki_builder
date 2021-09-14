import os
import logging
import sys
sys.path.append("../")
from typing import Tuple, List, Dict
from tqdm import tqdm
from collections import defaultdict

import dgl
import numpy as np
import torch
from dgl import DGLGraph

import utils
import data.utils as data_utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def prepare_graph(
        doc_path: str,
        cache_dir: str, 
        feature_type: List[str], 
        lang: str = "en",
        par_num: List[int] = None, 
        n_clusters: int = None,
    ) -> Tuple[DGLGraph, Dict[str, int], int]:
    # cached files
    os.makedirs(cache_dir, exist_ok=True)
    word2word_path = os.path.join(cache_dir, "word2word.pck")
    doc_info_path = os.path.join(cache_dir, "doc_info.pck")
    doc_concept_info_path = os.path.join(cache_dir, "doc_concept_info.pck")

    # loading nodes and edges
    num_nodes_dict, data_dict = {}, {}
    nodes, edges = defaultdict(lambda : {}), defaultdict(lambda : {})
    
    D, D_feat, D_label, D_mask, doc_content = utils.cache_to_path(doc_info_path, get_document, doc_path, lang=lang, cache_dir=cache_dir)
    nodes["doc"]["train_mask"] = torch.tensor(D_mask["train_mask"], dtype=torch.bool)
    nodes["doc"]["val_mask"] = torch.tensor(D_mask["val_mask"], dtype=torch.bool)
    nodes["doc"]["test_mask"] = torch.tensor(D_mask["test_mask"], dtype=torch.bool)
    nodes["doc"]["label"] = torch.tensor(D_label, dtype=torch.long)
    nodes["doc"]["feat"] = torch.tensor(D_feat, dtype=torch.float32)
    num_nodes_dict["doc"] = len(D)
    num_classes = nodes["doc"]["label"].shape[1]

    if "word" in feature_type:
        W, W_feat, DvsW_weight, WvsW_weight = utils.cache_to_path(word2word_path, get_document_word, doc_content, lang=lang, cache_dir=cache_dir)
        DvsW = DvsW_weight.nonzero()
        WvsW = WvsW_weight.nonzero()
        data_dict.update({
            ("word", "word#relate", "word"): WvsW,
            ("doc", "word#have", "word"): DvsW,
            ("word", "word#in", "doc"): (DvsW[1], DvsW[0]),
        })
        num_nodes_dict["word"] = len(W)
        nodes["word"]["feat"] = torch.tensor(W_feat, dtype=torch.float32)
        edges["word#relate"]["weight"] = torch.tensor(np.asarray(WvsW_weight[WvsW]).squeeze(), dtype=torch.float32)
        edges["word#have"]["weight"] = edges["word#in"]["weight"] = torch.tensor(np.asarray(DvsW_weight[DvsW]).squeeze(), dtype=torch.float32)

    if "doc_cluster" in feature_type:
        Cl_feat, DvsCl, ClvsCl, DvsCl_weight, ClvsCl_weight = get_document_cluster(D, D_feat, n_clusters=n_clusters)
        data_dict.update({
            ("cluster", "cluster#connect", "cluster"): ClvsCl,
            ("doc", "cluster#form", "cluster"): DvsCl,
            ("cluster", "cluster#in", "doc"): (DvsCl[1], DvsCl[0]),
        })
        num_nodes_dict["cluster"] = len(Cl_feat)
        nodes["cluster"]["feat"] = torch.tensor(Cl_feat, dtype=torch.float32)
        edges["cluster#connect"]["weight"] = torch.tensor(ClvsCl_weight[ClvsCl], dtype=torch.float32)
        edges["cluster#form"]["weight"] = edges["cluster#in"]["weight"] = torch.tensor(DvsCl_weight, dtype=torch.float32)

    if "concept" in feature_type:
        C, C_feat, DvsC, CvsC = utils.cache_to_path(doc_concept_info_path, get_document_concept, D, par_num=par_num, cache_dir=cache_dir)
        data_dict.update({
            ("doc", "concept#have", "concept"): DvsC,
            ("concept", "concept#in", "doc"): (DvsC[1], DvsC[0]),
            ("concept", "concept#relate", "concept"): CvsC,
            ("concept", "rev-concept#relate", "concept"): (CvsC[1], CvsC[0]),
        })
        num_nodes_dict["concept"] = len(C)
        nodes["concept"]["feat"] = torch.tensor(C_feat, dtype=torch.float32)

    # add self-loop
    for ntype, num_nodes in num_nodes_dict.items():
        data_dict.update({
            (ntype, f"{ntype}#self-loop", ntype): (range(num_nodes), range(num_nodes))
        })

    # create heterogeneous graph
    graph = dgl.heterograph(data_dict=data_dict, num_nodes_dict=num_nodes_dict)
    for ntype in nodes:
        for dtype in nodes[ntype]:
            graph.nodes[ntype].data[dtype] = nodes[ntype][dtype]
    for etype in edges:
        for dtype in edges[etype]:
            graph.edges[etype].data[dtype] = edges[etype][dtype]
    logging.info(graph)
    return graph.to(device), D, num_classes
 
def get_document(doc_path: str, lang: str = "en", cache_dir: str = "data/cache"):
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
    utils.dump_json(label_encoder, os.path.join(cache_dir, "doc_label_encoder.json"))
    D_label = utils.get_multihot_encoding(D_label, label_encoder)
    D_feat = utils.get_bert_features(D_feat, max_length=512, lang=lang)

    return D, D_feat, D_label, D_mask, doc_content

def get_document_word(doc_content: List[str], lang: str = "en", cache_dir: str = "data/cache"):
    doc_content = utils.normalize_text(doc_content, lang=lang, cache_dir=cache_dir)
    DvsW_weight, W = utils.get_tfidf_score(doc_content, lang=lang, cache_dir=cache_dir)
    sorted_words = sorted(W, key=W.get)
    W_feat = utils.get_word_embedding(sorted_words, corpus=doc_content, cache_dir=cache_dir)
    WvsW_weight = utils.get_pmi(doc_content, vocab=sorted_words, window_size=20)
    return W, W_feat, DvsW_weight, WvsW_weight

def get_document_cluster(D: Dict[str, int], D_feat: List, n_clusters: int = 100):
    Cl_feat, cluster_assignment, DvsCl_weight, ClvsCl_weight = utils.get_kmean_matrix(D_feat, num_cluster_list=[n_clusters])
    DvsCl = (np.array(range(len(D))), cluster_assignment)
    ClvsCl = ClvsCl_weight.nonzero()
    return Cl_feat, DvsCl, ClvsCl, DvsCl_weight, ClvsCl_weight

def get_document_concept(D: Dict[str, int], par_num: List[int], cache_dir: str = "data/cache"):
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
    C_feat = utils.get_phrase_embedding(C_feat)
    # C_feat = utils.get_bert_features(C_feat, max_length=32)

    DvsC = ([], [])
    CvsC = ([], [])
    for u, v in DvsC_edges:
        DvsC[0].append(D[u])
        DvsC[1].append(C[v])
    for u, v in CvsC_edges:
        CvsC[0].append(C[u])
        CvsC[1].append(C[v])

    return C, C_feat, DvsC, CvsC