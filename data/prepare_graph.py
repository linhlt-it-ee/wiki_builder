import os
import logging
import sys
sys.path.append("../")
from typing import Tuple, List, Dict, Callable
from tqdm import tqdm
from collections import defaultdict

import dgl
import numpy as np
import networkx as nx
import torch
from dgl import DGLGraph
import utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def prepare_graph(data_dir: str, feature_type: str, par_num: List[int] = None, return_graph: bool = False) -> Tuple[DGLGraph, Dict, Dict, int]:
    doc_path = os.path.join(data_dir, "merge_data.ndjson")
    doc_label_path = os.path.join(data_dir, "doc_label_encoder.json")
    cache_dir = os.path.join(data_dir, "cache")
    doc_mention_path = os.path.join(cache_dir, "doc_mention.json")
    mention_concept_path = os.path.join(cache_dir, "concept_links.json")
    concept_path = os.path.join(cache_dir, "concept_labels.json")
    # saved path
    word2word_path = os.path.join(cache_dir, "word2word.pck")
    doc_info_path = os.path.join(cache_dir, "doc_info.pck")
    doc_concept_info_path = os.path.join(cache_dir, "doc_concept_info.pck")

    # loading nodes and edges
    num_nodes_dict, data_dict = {}, {}
    nodes, edges = defaultdict(lambda : {}), defaultdict(lambda : {})
    
    # document features
    D, D_feat, D_label, D_mask, doc_content = utils.cache_to_path(doc_info_path, get_document, doc_path, doc_label_path)
    num_nodes_dict["doc"] = len(D)
    nodes["doc"]["train_mask"] = torch.tensor(D_mask["train_mask"], dtype=torch.bool)
    nodes["doc"]["val_mask"] = torch.tensor(D_mask["val_mask"], dtype=torch.bool)
    nodes["doc"]["test_mask"] = torch.tensor(D_mask["test_mask"], dtype=torch.bool)
    nodes["doc"]["label"] = torch.tensor(D_label, dtype=torch.long)
    nodes["doc"]["feat"] = torch.tensor(D_feat, dtype=torch.float32)

    if return_graph:
        return get_document_concept(D, concept_path, doc_mention_path, mention_concept_path, par_num, True)
    
    # getting other nodes
    if "concept" in feature_type:
        C, C_feat, DvsC, CvsC = utils.cache_to_path(doc_concept_info_path, get_document_concept, D, concept_path, doc_mention_path, mention_concept_path, par_num)
        data_dict.update({
            ("doc", "concept#have", "concept"): DvsC,
            ("concept", "concept#in", "doc"): (DvsC[1], DvsC[0]),
            ("concept", "concept#relate", "concept"): CvsC,
            ("concept", "rev-concept#relate", "concept"): (CvsC[1], CvsC[0]),
        })
        num_nodes_dict["concept"] = len(C)
        nodes["concept"]["feat"] = torch.tensor(C_feat, dtype=torch.float32)
        
    if "word" in feature_type:
        W, W_feat, DvsW, WvsW, DvsW_weight, WvsW_weight = get_document_word(doc_content, word2word_path)
        rev_DvsW_weight = DvsW_weight.transpose()
        rev_DvsW = rev_DvsW_weight.nonzero()
        rev_WvsW_weight = WvsW_weight.transpose()
        rev_WvsW = rev_WvsW_weight.nonzero()
        data_dict.update({
            ("doc", "word#have", "word"): DvsW,
            ("word", "word#in", "doc"): rev_DvsW,
            ("word", "word#relate", "word"): WvsW,
            ("word", "rev-word#relate", "word"): rev_WvsW,
        })
        num_nodes_dict["word"] = len(W)
        nodes["word"]["feat"] = torch.tensor(W_feat, dtype=torch.float32)
        edges["word#have"]["weight"] = torch.tensor(np.asarray(DvsW_weight[DvsW]).squeeze(), dtype=torch.float32)
        edges["word#in"]["weight"] = torch.tensor(np.asarray(rev_DvsW_weight[rev_DvsW]).squeeze(), dtype=torch.float32)
        edges["word#relate"]["weight"] = torch.tensor(np.asarray(WvsW_weight[WvsW]).squeeze(), dtype=torch.float32)
        edges["rev-word#relate"]["weight"] = torch.tensor(np.asarray(rev_WvsW_weight[rev_WvsW]).squeeze(), dtype=torch.float32)

    if "cluster" in feature_type:
        Cl_feat, DvsCl, ClvsCl, DvsCl_weight, ClvsCl_weight = get_document_cluster(D, D_feat)
        data_dict.update({
            ("doc", "cluster#form", "cluster"): DvsCl,
            ("cluster", "cluster#connect", "cluster"): ClvsCl,
            ("cluster", "cluster#in", "doc"): (DvsCl[1], DvsCl[0]),
        })
        num_nodes_dict["cluster"] = len(Cl_feat)
        nodes["cluster"]["feat"] = torch.tensor(Cl_feat, dtype=torch.float32)
        edges["cluster#form"]["weight"] = torch.tensor(DvsCl_weight, dtype=torch.float32)
        edges["cluster#connect"]["weight"] = torch.tensor(ClvsCl_weight[ClvsCl], dtype=torch.float32)

    # create heterogeneous graph
    graph = dgl.heterograph(data_dict=data_dict, num_nodes_dict=num_nodes_dict)
    num_classes = len(utils.load_json(doc_label_path))
    for ntype in nodes:
        for dtype in nodes[ntype]:
            graph.nodes[ntype].data[dtype] = nodes[ntype][dtype]
    for etype in edges:
        for dtype in edges[etype]:
            graph.edges[etype].data[dtype] = edges[etype][dtype]
    logging.info(graph)

    return graph.to(device), D, num_classes

def get_document(doc_path: str, doc_label_path: str):
    doc_labels = utils.load_json(doc_label_path)
    docs = [doc for doc in utils.load_ndjson(doc_path)]
    D = {did: id for id, did in enumerate(sorted(doc["id"] for doc in docs))}
    doc_content = [None] * len(D)
    D_feat = [None] * len(D)
    D_label = [None] * len(D)
    D_feat = [None] * len(D)
    D_mask = {data_type: [None] * len(D) for data_type in ("train_mask", "val_mask", "test_mask")}

    for doc in tqdm(docs, desc="Encoding documents"):
        id = D.get(doc["id"], None)
        if id is not None:
            doc_content[id] = doc["content"]
            D_feat[id] = doc["desc"]
            D_label[id] = utils.get_onehot(doc["labels"], doc_labels)
            D_mask["train_mask"][id] = doc["is_train"]
            D_mask["val_mask"][id] = doc["is_dev"]
            D_mask["test_mask"][id] = doc["is_test"]
    D_feat = utils.get_bert_features(D_feat, max_length=512)

    return D, D_feat, D_label, D_mask, doc_content

def get_document_concept(D: Dict, concept_path: str, doc_mention_path: str, mention_concept_path: str, par_num: List[int], return_graph: bool = False):
    doc_mention = utils.load_json(doc_mention_path)
    valid_mentions = set()
    for x in doc_mention.values():
        valid_mentions.update(x)

    # create concept graph
    mention_concept = utils.load_json(mention_concept_path)
    CvsC_graph = nx.DiGraph()
    for x in mention_concept.values():
        if x["label"] in valid_mentions:
            for parent in x["parents"]:
                if parent["level"] == 1:
                    continue
                path = parent["path"].split(" >> ")
                nx.add_path(CvsC_graph, path)

    C = set()       # not include name mentions
    C_all = set()   # include name mentions
    C_par = {}
    children = set(x for x in mention_concept.keys() if x in CvsC_graph.nodes)
    for level, parlevel_num in enumerate(par_num, start=1):
        cnt = defaultdict(lambda : 0)
        C_all.update(children)
        for child in children:
            for par in CvsC_graph.successors(child):
                if par not in C_all:    # disable nodes with different levels
                    cnt[par] += 1
        # discard parents with less than 2 children
        children = [k for k, v in sorted(cnt.items(), key=lambda x : (x[1], x[0]), reverse=True) if v >= 2]
        children = children[:parlevel_num]
        logging.info(f"Extracting {parlevel_num} parents level {level} with most children, got {len(children)}")
        C.update(children)
        C_par[level] = children
    C_all.update(children)
    if return_graph:
        return nx.DiGraph(CvsC_graph.subgraph(C_all)), C_par

    CvsC_graph = nx.DiGraph(CvsC_graph.subgraph(C))
    logging.info(f"CvsC graph: {CvsC_graph}")
    logging.info(f"C size: {len(C)}")
    
    # mapping name_mentions and their IDs to connect a document and its mentioned concepts
    mention_ids = defaultdict(set)
    for mid, mention_info in mention_concept.items():
        for label in set(x.lower() for x in mention_info["name_mention"]):  # match normalized nouns
            mention_ids[label].add(mid)

    DvsC_graph = nx.DiGraph()
    for did, mentions in doc_mention.items():
        DvsC_graph.add_node(did)
        # traverse meanings of a name_mention in a document
        for label in mentions:
            for mid in mention_ids[label]:
                # consider concept level 1 of each meaning
                for concept in mention_concept[mid]["parents"]:
                    if concept["level"] == 1 and concept["id"] in CvsC_graph.nodes:
                        DvsC_graph.add_edge(did, concept["id"])
    logging.info(f"DvsC graph: {DvsC_graph}")
    logging.info(f"D size: {len(D)}")

    C = {cid: id for id, cid in enumerate(sorted(C))}
    concepts = utils.load_json(concept_path)
    C_feat = [None] * len(C)
    for cid, label in tqdm(concepts.items(), "Encoding concepts"):
        id = C.get(cid, None)
        if id is not None:
            C_feat[id] = label
    C_feat = utils.get_bert_features(C_feat, max_length=32)

    # flatten
    DvsC = ([], [])
    CvsC = ([], [])
    for u, v in DvsC_graph.edges:
        DvsC[0].append(D[u])
        DvsC[1].append(C[v])
    for u, v in CvsC_graph.edges:
        CvsC[0].append(C[u])
        CvsC[1].append(C[v])

    return C, C_feat, DvsC, CvsC

def get_document_word(doc_content: List[str], word2word_path: str):
    if os.path.exists(word2word_path):
        W, W_feat, DvsW_weight, WvsW_weight = utils.load(word2word_path)
    else:
        doc_content = utils.normalize_text(doc_content)
        DvsW_weight, W = utils.get_tfidf_score(doc_content)
        sorted_words = [None] * len(W)
        for k, v in W.items():
            sorted_words[v] = k
        W_feat = utils.get_word_embedding(sorted_words, corpus=doc_content)
        WvsW_weight = utils.get_pmi(doc_content, vocab=sorted_words, window_size=20)
        utils.dump((W, W_feat, DvsW_weight, WvsW_weight), word2word_path)

    DvsW = DvsW_weight.nonzero()
    WvsW = WvsW_weight.nonzero()

    return W, W_feat, DvsW, WvsW, DvsW_weight, WvsW_weight

def get_document_cluster(D: Dict, D_feat: List):
    Cl_feat, cluster_assignment, DvsCl_weight, ClvsCl_weight = utils.get_kmean_matrix(D_feat, num_cluster_list=[100])
    DvsCl = (np.array(range(len(D))), cluster_assignment)
    ClvsCl = np.tril(ClvsCl_weight).nonzero()
    return Cl_feat, DvsCl, ClvsCl, DvsCl_weight, ClvsCl_weight
